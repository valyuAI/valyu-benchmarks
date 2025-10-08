#!/usr/bin/env node

// ============================================================================
// IMPORTS
// ============================================================================

import 'dotenv/config';
import { generateText, stepCountIs, tool } from 'ai';
import { google } from '@ai-sdk/google';
import { z } from 'zod';
import { Valyu, SearchType as ValyuSearchSDKType } from 'valyu-js';
import { Exa } from 'exa-js';

// ============================================================================
// ENVIRONMENT VARIABLES
// ============================================================================

const GOOGLE_GENERATIVE_AI_API_KEY = process.env.GOOGLE_GENERATIVE_AI_API_KEY;
const VALYU_API_KEY = process.env.VALYU_API_KEY;
const EXA_API_KEY = process.env.EXA_API_KEY;
const PARALLEL_API_KEY = process.env.PARALLEL_API_KEY;
const SERPAPI_KEY = process.env.SERPAPI_KEY;

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

const MAX_TOOL_CALLS = 5;
const GEMINI_MODEL = 'gemini-2.5-pro';

// ============================================================================
// API CLIENT INITIALIZATION
// ============================================================================

const exa = new Exa(EXA_API_KEY);

// ============================================================================
// TOOL DEFINITIONS
// ============================================================================

/**
 * Valyu Deep Search Tool
 * Searches using Valyu across all available sources including academic papers, finance books, SEC filings, web content, and market data.
 */
export const valyuDeepSearchTool = tool({
  description:
    "Search Valyu for comprehensive information across all sources including academic papers, finance books, SEC filings, web content, and market data. Use for specific, up-to-date information across various domains.",
  inputSchema: z.object({
    query: z
      .string()
      .describe(
        'Detailed search query (e.g., "latest advancements in AI for healthcare" or "current price of Bitcoin").'
      ),
  }),
  execute: async ({ query }: { query: string; }) => {
    if (!VALYU_API_KEY) {
      console.error("[ValyuDeepSearchTool] VALYU_API_KEY is not set.");
      return JSON.stringify({
        success: false,
        error: "Valyu API key not configured.",
        results: [],
      });
    }

    const valyu = new Valyu(VALYU_API_KEY);

    try {
      console.error(`[ValyuDeepSearchTool] Searching: "${query}"`);

      const response = await valyu.search(
        query,
        {
          searchType: "all" as ValyuSearchSDKType,
          maxNumResults: 3,
          maxPrice: 5000.0,
          responseLength: "large",
        }
      );

      if (!response.success) {
        console.error("[ValyuDeepSearchTool] API Error:", response.error);
        return JSON.stringify({
          success: false,
          error: response.error || "Valyu API request failed.",
          query,
          results: [],
        });
      }

      console.error(`[ValyuDeepSearchTool] Search completed successfully - Received ${response.results?.length || 0} results`);

      return JSON.stringify({
        success: true,
        query,
        results: response.results || [],
        tx_id: response.tx_id,
      });
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error.";
      console.error("[ValyuDeepSearchTool] Exception:", errorMessage);
      return JSON.stringify({
        success: false,
        error: errorMessage,
        query,
        results: [],
      });
    }
  },
});

/**
 * Exa Search Tool
 * Searches the web using Exa with livecrawl for up-to-date information
 */
export const exaSearchTool = tool({
  description: 'Search the web for up-to-date information using Exa',
  inputSchema: z.object({
    query: z.string().min(1).max(100).describe('The search query'),
  }),
  execute: async ({ query }) => {
    try {
      if (!EXA_API_KEY) {
        throw new Error('EXA_API_KEY environment variable is not set');
      }

      console.error(`[ExaSearchTool] Searching: "${query}"`);

      const { results } = await exa.searchAndContents(query, {
        livecrawl: 'always',
        numResults: 3,
      });

      console.error(`[ExaSearchTool] Success - ${results.length} results`);

      return results.map(result => ({
        title: result.title,
        url: result.url,
        content: result.text?.slice(0, 100000),
        publishedDate: result.publishedDate,
      }));
    } catch (error) {
      console.error('[ExaSearchTool] Error:', error);
      return {
        error: error instanceof Error ? error.message : 'Unknown error occurred',
        query,
      };
    }
  },
});

/**
 * Google Search Tool
 * Searches Google using SerpAPI for organic search results
 */
export const googleSearchTool = tool({
  description: 'Search Google for up-to-date information using SerpAPI - returns organic search results',
  inputSchema: z.object({
    query: z.string().min(1).max(200).describe('The search query'),
    numResults: z.number().min(1).max(20).optional().describe('Number of results to return (default: 10)'),
    language: z.string().optional().describe('Language code (default: "en")'),
    country: z.string().optional().describe('Country code (default: "us")'),
  }),
  execute: async ({ query, numResults = 10, language = 'en', country = 'us' }) => {
    try {
      if (!SERPAPI_KEY) {
        throw new Error('SERPAPI_KEY environment variable is not set');
      }

      const url = new URL('https://serpapi.com/search.json');
      url.searchParams.append('engine', 'google');
      url.searchParams.append('api_key', SERPAPI_KEY);
      url.searchParams.append('q', query);
      url.searchParams.append('hl', language);
      url.searchParams.append('gl', country);
      url.searchParams.append('num', numResults.toString());

      console.error(`[GoogleSearchTool] Searching: "${query}"`);

      const response = await fetch(url.toString());

      if (!response.ok) {
        throw new Error(`SerpAPI request failed: ${response.status} ${response.statusText}`);
      }

      const json = await response.json();

      console.error(`[GoogleSearchTool] Success - ${json.organic_results?.length || 0} results`);

      const organicResults = json.organic_results || [];
      return organicResults.slice(0, 3);
    } catch (error) {
      console.error('[GoogleSearchTool] Error:', error);
      return {
        error: error instanceof Error ? error.message : 'Unknown error occurred during search',
        query,
        search_parameters: {
          q: query,
          hl: language,
          gl: country,
          num: numResults
        }
      };
    }
  },
});

/**
 * Parallel AI Search Tool
 * Searches the web using Parallel AI for comprehensive results
 */
export const parallelSearchTool = tool({
  description: 'Search the web using Parallel AI for comprehensive and optimized results',
  inputSchema: z.object({
    query: z.string().max(5000).describe('The search query to use as the objective for web research'),
  }),
  execute: async ({ query }) => {
    try {
      if (!PARALLEL_API_KEY) {
        throw new Error('PARALLEL_API_KEY environment variable is not set');
      }

      console.error(`[ParallelSearchTool] Searching: "${query}"`);

      const response = await fetch('https://api.parallel.ai/v1beta/search', {
        method: 'POST',
        headers: {
          'x-api-key': PARALLEL_API_KEY,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          objective: query,
          processor: 'base',
          max_results: 3,
          max_chars_per_result: 100000,
          search_queries: []
        })
      });

      if (!response.ok) {
        throw new Error(`Parallel API request failed: ${response.status} ${response.statusText}`);
      }

      const json = await response.json();
      console.error(`[ParallelSearchTool] Success - ${json.results?.length || 0} results`);

      return {
        search_id: json.search_id,
        results: json.results || []
      };
    } catch (error) {
      console.error('[ParallelSearchTool] Error:', error);
      return {
        error: error instanceof Error ? error.message : 'Unknown error occurred during search',
        objective: query,
        search_parameters: {
          objective: query,
          processor: 'base',
          max_results: 5,
          max_chars_per_result: 100000
        }
      };
    }
  },
});

// ============================================================================
// TOOL SELECTION
// ============================================================================

/**
 * Get tools based on the selected tool type
 */
function getToolsByType(toolType: string) {
  switch (toolType.toLowerCase()) {
    case 'valyu':
      console.error('[AgentSearch] Using Valyu Search');
      return { valyuSearch: valyuDeepSearchTool };

    case 'google':
      console.error('[AgentSearch] Using Google Search');
      return { googleSearch: googleSearchTool };

    case 'exa':
      console.error('[AgentSearch] Using Exa Search');
      return { exaSearch: exaSearchTool };

    case 'parallel':
      console.error('[AgentSearch] Using Parallel AI Search');
      return { parallelSearch: parallelSearchTool };

    default:
      console.error(`[AgentSearch] Unknown tool type "${toolType}", defaulting to Valyu`);
      return { valyuSearch: valyuDeepSearchTool };
  }
}

// ============================================================================
// SYSTEM PROMPTS
// ============================================================================

/**
 * Generate system prompt for the query
 */
function getSystemPrompt(question: string, maxToolCalls: number): string {
  const currentDate = new Date().toISOString().split('T')[0];

  return `You are a helpful assistant. Today's date is ${currentDate}.

You have access to a web search tool that allows you to search the internet for up-to-date information. You may use this tool up to ${maxToolCalls} times per question, so use each tool call wisely.

When you use the web search tool, make sure your search queries are well contextualized and specifically tailored to retrieve the most relevant information for answering the user's question. Think carefully about what keywords, facts, or context will make your search most effective for a web search engine.

Do not waste tool calls on vague or overly broad queries. Instead, break down the question if needed and craft each search query to maximize the chance of finding the exact information required. Consider what a human would type into a search engine to get the best results.

Do not spend time being skeptical or analyzing the validity of the question—assume the question is genuine and focus on providing a clear, direct, and informative answer. If the question is ambiguous, use your best judgment to interpret it and search for the most likely intended information.

After gathering enough information from your web searches, synthesize a complete and accurate answer for the user. If the information is not available, clearly state that you could not find an answer.

Question: ${question}

Begin by thinking about what information is needed to answer the question, then use the web search tool with well-crafted, context-rich queries (up to ${maxToolCalls} times) to find that information. Provide a clear and concise answer based on your findings. Remember: each tool call is valuable, so make every search count.`;
}

// ============================================================================
// MAIN QUERY FUNCTION
// ============================================================================

/**
 * Main query function for benchmarking
 */
export async function queryForBenchmark(
  question: string,
  questionType: string = 'financial',
  toolType: string = 'valyu'
): Promise<string> {
  try {
    // Check for required Gemini API key
    if (!GOOGLE_GENERATIVE_AI_API_KEY) {
      throw new Error('GOOGLE_GENERATIVE_AI_API_KEY environment variable is not set. Please get your API key from https://aistudio.google.com/app/apikey');
    }

    const systemPrompt = getSystemPrompt(question, MAX_TOOL_CALLS);
    const tools = getToolsByType(toolType);

    const { text, response } = await generateText({
      model: google(GEMINI_MODEL),
      tools: tools,
      prompt: systemPrompt,
      stopWhen: stepCountIs(MAX_TOOL_CALLS + 1),
    });

    // Extract tool outputs from conversation messages
    const toolOutputs: any[] = [];
    response.messages.forEach((message, index) => {
      if (message.role === 'tool') {
        try {
          const messageContent = Array.isArray(message.content) ? message.content[0] : message.content;
          const toolName = (messageContent as any)?.toolName;
          const toolCallId = (messageContent as any)?.toolCallId;
          const output = (messageContent as any)?.output;

          if (output && toolName) {
            const toolOutput: any = {
              toolName,
              toolCallId,
              messageIndex: index + 1,
            };

            // Parse tool-specific outputs
            if (toolName === 'valyuSearch') {
              try {
                const parsedValue = JSON.parse(output.value);
                toolOutput.output = {
                  success: parsedValue.success || false,
                  query: parsedValue.query || '',
                  resultCount: parsedValue.results ? parsedValue.results.length : 0,
                  results: parsedValue.results || [],
                  tx_id: parsedValue.tx_id || ''
                };
              } catch (e) {
                toolOutput.output = {
                  success: false,
                  error: 'Failed to parse Valyu response',
                  rawOutput: output.value
                };
              }
            } else if (toolName === 'exaSearch') {
              try {
                const results = output.value;
                toolOutput.output = {
                  success: true,
                  resultCount: Array.isArray(results) ? results.length : 0,
                  results: results || []
                };
              } catch (e) {
                toolOutput.output = {
                  success: false,
                  error: 'Failed to parse Exa response',
                  rawOutput: output.value
                };
              }
            } else if (toolName === 'googleSearch') {
              try {
                const results = output.value;
                if (results && typeof results === 'object' && results.error) {
                  toolOutput.output = {
                    success: false,
                    error: results.error,
                    query: results.query || '',
                    resultCount: 0
                  };
                } else {
                  toolOutput.output = {
                    success: true,
                    resultCount: Array.isArray(results) ? results.length : 0,
                    results: results || []
                  };
                }
              } catch (e) {
                toolOutput.output = {
                  success: false,
                  error: 'Failed to parse Google search response',
                  rawOutput: output.value
                };
              }
            } else if (toolName === 'parallelSearch') {
              try {
                const response = output.value;
                if (response && response.error) {
                  toolOutput.output = {
                    success: false,
                    error: response.error,
                    objective: response.objective || '',
                    resultCount: 0
                  };
                } else {
                  toolOutput.output = {
                    success: true,
                    search_id: response.search_id || '',
                    resultCount: response.results ? response.results.length : 0,
                    results: response.results || []
                  };
                }
              } catch (e) {
                toolOutput.output = {
                  success: false,
                  error: 'Failed to parse Parallel search response',
                  rawOutput: output.value
                };
              }
            } else {
              toolOutput.output = output;
            }

            toolOutputs.push(toolOutput);
          }
        } catch (e) {
          toolOutputs.push({
            toolName: 'unknown',
            toolCallId: 'unknown',
            messageIndex: index + 1,
            output: {
              success: false,
              error: 'Failed to parse tool message',
              rawContent: JSON.stringify(message.content).slice(0, 200)
            }
          });
        }
      }
    });

    console.error(`[AgentSearch] Total messages: ${response.messages.length}`);
    console.error(`[AgentSearch] Tool outputs captured: ${toolOutputs.length}`);
    toolOutputs.forEach((toolOutput, index) => {
      console.error(`  Tool ${index + 1}: ${toolOutput.toolName} (${toolOutput.output.success ? 'Success' : 'Error'})`);
    });

    const benchmarkResult = {
      text: text,
      toolOutputs: toolOutputs,
      messageCount: response.messages.length
    };

    return JSON.stringify(benchmarkResult);
  } catch (error) {
    console.error('[AgentSearch] Error in queryForBenchmark:', error);
    throw error;
  }
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

async function main() {
  const args = process.argv.slice(2);

  if (args.includes('--benchmark') || args.includes('--query')) {
    const questionIndex = args.findIndex(arg => arg === '--question' || arg === '--query');
    const typeIndex = args.findIndex(arg => arg === '--type');
    const toolIndex = args.findIndex(arg => arg === '--tool');

    if (questionIndex === -1 || questionIndex + 1 >= args.length) {
      console.error('Error: --question parameter is required for benchmark mode');
      console.error('Usage: npx tsx query.ts --benchmark --question "Your question here" [--type financial] [--tool valyu]');
      process.exit(1);
    }

    const question = args[questionIndex + 1];
    const questionType = typeIndex !== -1 && typeIndex + 1 < args.length ? args[typeIndex + 1] : 'financial';
    const toolType = toolIndex !== -1 && toolIndex + 1 < args.length ? args[toolIndex + 1] : 'valyu';

    try {
      console.error(`[AgentSearch] Processing question: ${question}`);
      console.error(`[AgentSearch] Tool type: ${toolType}`);
      const response = await queryForBenchmark(question, questionType, toolType);
      console.log(response);
      console.error(`[AgentSearch] Query completed successfully`);
    } catch (error) {
      console.error(`[AgentSearch] Error: ${error}`);
      process.exit(1);
    }
  } else {
    console.error('Usage: npx tsx query.ts --benchmark --question "Your question here" [--type financial] [--tool valyu]');
    console.error('');
    console.error('Options:');
    console.error('  --question       The question to answer (required)');
    console.error('  --type           Question type (default: financial)');
    console.error('  --tool           Search tool to use: valyu, google, exa, parallel (default: valyu)');
    process.exit(1);
  }
}

// Run the script
const isMainModule = process.argv[1] && process.argv[1].endsWith('query.ts');
if (isMainModule) {
  main().catch(console.error);
}

export { main };
