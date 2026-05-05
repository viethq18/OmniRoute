import { register } from "../registry.ts";
import { FORMATS } from "../formats.ts";
import {
  DEFAULT_SAFETY_SETTINGS,
  tryParseJSON,
  cleanJSONSchemaForAntigravity,
} from "../helpers/geminiHelper.ts";
import { buildGeminiTools, sanitizeGeminiToolName } from "../helpers/geminiToolsSanitizer.ts";
import { capMaxOutputTokens } from "../../../src/lib/modelCapabilities.ts";
import { resolveGeminiThoughtSignature } from "../../services/geminiThoughtSignatureStore.ts";

function extractClaudeThoughtSignature(block: unknown): string | null {
  if (!block || typeof block !== "object") return null;
  const value = block as Record<string, unknown>;
  const candidates = [value.thoughtSignature, value.thought_signature, value.signature];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.length > 0) return candidate;
  }
  return null;
}

/**
 * Direct Claude → Gemini request translator.
 * Converts Claude Messages API body directly to Gemini format,
 * skipping the OpenAI hub intermediate step.
 */
export function claudeToGeminiRequest(model, body, stream) {
  const toolNameMap = new Map<string, string>();
  const sanitizeToolName = (name: string) =>
    sanitizeGeminiToolName(name, {
      toolNameMap,
    });
  const result: {
    model: string;
    contents: Array<Record<string, unknown>>;
    generationConfig: Record<string, unknown>;
    safetySettings: unknown;
    systemInstruction?: { role: string; parts: Array<{ text: string }> };
    tools?: Array<{
      functionDeclarations?: Array<Record<string, unknown>>;
      googleSearch?: Record<string, unknown>;
      googleSearchRetrieval?: Record<string, unknown>;
    }>;
    _toolNameMap?: Map<string, string>;
  } = {
    model: model,
    contents: [],
    generationConfig: {},
    safetySettings: DEFAULT_SAFETY_SETTINGS,
  };

  // ── Generation config ──────────────────────────────────────────
  if (body.temperature !== undefined) {
    result.generationConfig.temperature = body.temperature;
  }
  if (body.top_p !== undefined) {
    result.generationConfig.topP = body.top_p;
  }
  if (body.top_k !== undefined) {
    result.generationConfig.topK = body.top_k;
  }
  if (body.max_tokens !== undefined) {
    result.generationConfig.maxOutputTokens = capMaxOutputTokens(model, body.max_tokens);
  }

  // ── System instruction ─────────────────────────────────────────
  if (body.system) {
    let systemText;
    if (Array.isArray(body.system)) {
      systemText = body.system.map((s) => s.text || "").join("\n");
    } else {
      systemText = String(body.system);
    }
    if (systemText) {
      result.systemInstruction = {
        role: "system",
        parts: [{ text: systemText }],
      };
    }
  }

  // ── Build tool_use name lookup (for tool_result matching) ──────
  const toolUseNames = {};
  if (body.messages && Array.isArray(body.messages)) {
    for (const msg of body.messages) {
      if (msg.role === "assistant" && Array.isArray(msg.content)) {
        for (const block of msg.content) {
          if (block.type === "tool_use" && block.id && block.name) {
            toolUseNames[block.id] = sanitizeToolName(block.name);
          }
        }
      }
    }
  }

  // ── Convert messages ───────────────────────────────────────────
  if (body.messages && Array.isArray(body.messages)) {
    for (const msg of body.messages) {
      const parts = [];
      const toolUseClientSignatures = new Map<string, string | null>();
      let firstPersistedToolSignature: string | null = null;

      if (Array.isArray(msg.content)) {
        let latestThinkingSignature: string | null = null;

        for (const block of msg.content) {
          if (block.type === "thinking") {
            const signature = extractClaudeThoughtSignature(block);
            if (signature) {
              latestThinkingSignature = signature;
            }
            continue;
          }
          if (block.type !== "tool_use") continue;
          const candidateSignature =
            extractClaudeThoughtSignature(block) || latestThinkingSignature;
          toolUseClientSignatures.set(block.id, candidateSignature);
          const resolved = resolveGeminiThoughtSignature(block.id, candidateSignature);
          if (!firstPersistedToolSignature && resolved) {
            firstPersistedToolSignature = resolved;
          }
        }

        for (const block of msg.content) {
          switch (block.type) {
            case "text":
              if (block.text) parts.push({ text: block.text });
              break;

            case "thinking":
              // Preserve thinking blocks as thought parts
              if (block.thinking) {
                const signature = extractClaudeThoughtSignature(block);
                parts.push({
                  ...(signature ? { thoughtSignature: signature } : {}),
                  thought: true,
                  text: block.thinking,
                });
              }
              break;

            case "tool_use": {
              const signatureForToolCall = resolveGeminiThoughtSignature(
                block.id,
                toolUseClientSignatures.get(block.id)
              );
              const embeddedThoughtSignature =
                signatureForToolCall || firstPersistedToolSignature || undefined;

              parts.push({
                ...(embeddedThoughtSignature
                  ? {
                      thoughtSignature: embeddedThoughtSignature,
                    }
                  : {}),
                functionCall: {
                  id: block.id,
                  name: sanitizeToolName(block.name),
                  args: block.input || {},
                },
              });
              break;
            }

            case "tool_result": {
              let content = block.content;
              if (Array.isArray(content)) {
                content = content
                  .map((c) => (c.type === "text" ? c.text : JSON.stringify(c)))
                  .join("\n");
              }
              let parsedContent = tryParseJSON(content);
              if (parsedContent === null) {
                parsedContent = { result: content };
              } else if (typeof parsedContent !== "object") {
                parsedContent = { result: parsedContent };
              }
              parts.push({
                functionResponse: {
                  id: block.tool_use_id,
                  name: toolUseNames[block.tool_use_id] || "unknown",
                  response: { result: parsedContent },
                },
              });
              break;
            }

            case "image":
              // Base64 image → Gemini inlineData
              if (block.source?.type === "base64") {
                parts.push({
                  inlineData: {
                    mimeType: block.source.media_type,
                    data: block.source.data,
                  },
                });
              }
              break;
          }
        }
      } else if (typeof msg.content === "string" && msg.content) {
        parts.push({ text: msg.content });
      }

      if (parts.length > 0) {
        // Map Claude roles to Gemini roles
        const geminiRole = msg.role === "assistant" ? "model" : "user";

        // Gemini 3+ expects the signature on all functionCall parts in a tool-call
        // batch. If there is no real signature, we don't inject a fake one because
        // Gemini API strictly validates it and returns 400.
        if (geminiRole === "model") {
          // No operation needed since we no longer inject fake signatures.
        }

        result.contents.push({ role: geminiRole, parts });
      }
    }
  }

  // ── Convert tools ──────────────────────────────────────────────
  const geminiTools = buildGeminiTools(body.tools, {
    toolNameMap,
  });
  if (geminiTools) {
    result.tools = geminiTools;
  }

  // ── Thinking config ────────────────────────────────────────────
  // Priority: thinking.budget_tokens (Claude native) > output_config.effort (Claude Code).
  if (model.startsWith("gemma-4")) {
    // gemma-4 models returns - 400: Thinking budget is not supported for this model
  } else if (body.thinking?.type === "enabled" && body.thinking.budget_tokens) {
    result.generationConfig.thinkingConfig = {
      thinkingBudget: body.thinking.budget_tokens,
      includeThoughts: true,
    };
  } else if (typeof body.output_config?.effort === "string") {
    const effort = body.output_config.effort.toLowerCase();
    const effortBudgetMap: Record<string, number> = {
      none: 0,
      low: 1024,
      medium: 10240,
      high: 32768,
      max: 131072,
      xhigh: 131072,
    };
    const budget = effortBudgetMap[effort];
    if (budget !== undefined && budget > 0) {
      result.generationConfig.thinkingConfig = {
        thinkingBudget: budget,
        includeThoughts: true,
      };
    }
  }

  const changedToolNameMap = new Map(
    [...toolNameMap.entries()].filter(
      ([sanitizedName, originalName]) => sanitizedName !== originalName
    )
  );
  if (changedToolNameMap.size > 0) {
    result._toolNameMap = changedToolNameMap;
  }

  return result;
}

// Register direct path only for plain Gemini API.
// Gemini CLI / Antigravity require Cloud Code envelope wrapping,
// so they must use the existing hub path (Claude -> OpenAI -> target).
register(FORMATS.CLAUDE, FORMATS.GEMINI, claudeToGeminiRequest, null);
