#!/usr/bin/env node
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

      console.error(`[ParallelSearchTool] Searching for: "${query}"`);

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
      console.error(`[ParallelSearchTool] Search completed successfully`);

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
          max_results: 3,
          max_chars_per_result: 100000
        }
      };
    }
  },
});

/**
 * Exa Search Tool
 * Searches the web using Exa for up-to-date information with live crawling
 */
export const exaSearchTool = tool({
  description: 'Search the web for up-to-date information',
  inputSchema: z.object({
    query: z.string().min(1).max(100).describe('The search query'),
  }),
  execute: async ({ query }) => {
    console.error(`[ExaSearchTool] Searching for: "${query}"`);

    const { results } = await exa.searchAndContents(query, {
      livecrawl: 'always',
      numResults: 3,
      maxChars: 100000,
    });

    console.error(`[ExaSearchTool] Search completed successfully`);

    return results.map(result => ({
      title: result.title,
      url: result.url,
      content: result.text?.slice(0, 100000),
      publishedDate: result.publishedDate,
    }));
  },
});

/**
 * Google Search Tool (SerpAPI)
 * Searches Google using SerpAPI for organic search results
 */
export const googleSearchTool = tool({
  description: 'Search Google for up-to-date information using SerpAPI - returns organic search results',
  inputSchema: z.object({
    query: z.string().min(1).max(200).describe('The search query'),
    numResults: z.number().min(1).max(3).optional().describe('Number of results to return (default: 10)'),
    language: z.string().optional().describe('Language code (default: "en")'),
    country: z.string().optional().describe('Country code (default: "us")'),
  }),
  execute: async ({ query, numResults = 10, language = 'en', country = 'us' }) => {
    try {
      if (!SERPAPI_KEY) {
        throw new Error('SERPAPI_KEY environment variable is not set');
      }

      console.error(`[GoogleSearchTool] Searching for: "${query}"`);

      const url = new URL('https://serpapi.com/search.json');
      url.searchParams.append('engine', 'google');
      url.searchParams.append('api_key', SERPAPI_KEY);
      url.searchParams.append('q', query);
      url.searchParams.append('hl', language);
      url.searchParams.append('gl', country);
      url.searchParams.append('num', numResults.toString());

      const response = await fetch(url.toString());

      if (!response.ok) {
        throw new Error(`SerpAPI request failed: ${response.status} ${response.statusText}`);
      }

      const json = await response.json();
      console.error(`[GoogleSearchTool] Search completed successfully`);

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
 * Valyu Deep Search Tool
 * Searches Valyu for real-time academic papers, web content, market data, etc.
 */
export const valyuDeepSearchTool = tool({
  description: 'Search Valyu for real-time academic papers, web content, market data, etc. Use for specific, up-to-date information across various domains.',
  inputSchema: z.object({
    query: z.string().describe('Detailed search query (e.g., "latest advancements in AI for healthcare" or "current price of Bitcoin").'),
  }),
  execute: async ({ query }: { query: string }) => {
    if (!VALYU_API_KEY) {
      console.error('[ValyuDeepSearchTool] VALYU_API_KEY is not set');
      return JSON.stringify({
        success: false,
        error: 'Valyu API key not configured.',
        results: [],
      });
    }

    console.error(`[ValyuDeepSearchTool] Searching for: "${query}"`);

    const valyu = new Valyu(VALYU_API_KEY);

    try {
      const response = await valyu.search(query, {
        searchType: 'all' as ValyuSearchSDKType,
        maxNumResults: 3,
        maxPrice: 5000.0,
        responseLength: 'large',
      });

      if (!response.success) {
        console.error('[ValyuDeepSearchTool] API Error:', response.error);
        return JSON.stringify({
          success: false,
          error: response.error || 'Valyu API request failed.',
          query,
          results: [],
        });
      }

      console.error(`[ValyuDeepSearchTool] Search completed successfully - Received ${response.results?.length || 0} results`);

      return JSON.stringify({
        success: true,
        query,
        results: response.results || [],
        metadata: {
          totalResults: response.results?.length || 0,
        },
      });
    } catch (error) {
      console.error('[ValyuDeepSearchTool] Error:', error);
      return JSON.stringify({
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred during search.',
        query,
        results: [],
      });
    }
  },
});

// ============================================================================
// SYSTEM PROMPTS
// ============================================================================

const createFinanceSystemPrompt = (question: string, currentDate: string, maxToolCalls: number) => `You are a financial research assistant with access to search tools. Today's date is ${currentDate}.

## Your Task
Answer the following financial question accurately and comprehensively: ${question}

## Available Tools
You have access to search tools that can retrieve financial data, company information, market data, and other relevant sources. Use these tools to gather the information needed to answer the question.

## Research Approach
- Use your tools iteratively to gather comprehensive information
- Build each query based on findings from previous searches - reference specific data points, figures, or details you've discovered to make subsequent queries more targeted
- You may use up to ${maxToolCalls} tool calls to thoroughly research the question
- Provide your final answer once you have sufficient information to respond accurately

## Final Answer
Provide a clear, complete answer that:
- Directly addresses the question asked
- Includes relevant data, numbers, and dates
- Cites sources when possible
- Is based on the information gathered through your research`;

const createMedicalSystemPrompt = (question: string, currentDate: string, maxToolCalls: number) => `You are a medical research assistant with access to search tools. Today's date is ${currentDate}.

## Your Task
Answer the following medical question accurately and comprehensively: ${question}

## Available Tools
You have access to search tools that can retrieve medical data, clinical information, research studies, and other relevant sources. Use these tools to gather the information needed to answer the question.

## Research Approach
- Use your tools iteratively to gather comprehensive information
- Build each query based on findings from previous searches - reference specific data points, figures, or details you've discovered to make subsequent queries more targeted
- You may use up to ${maxToolCalls} tool calls to thoroughly research the question
- Provide your final answer once you have sufficient information to respond accurately

## Final Answer
Provide a clear, complete answer that:
- Directly addresses the question asked
- Includes relevant data, numbers, and dates
- Cites sources when possible
- Is based on the information gathered through your research`;

const createEconomicSystemPrompt = (question: string, currentDate: string, maxToolCalls: number) => `You are an economics research assistant with access to search tools. Today's date is ${currentDate}.

## Your Task
Answer the following economics question accurately and comprehensively: ${question}

## Available Tools
You have access to search tools that can retrieve economic data, economic indicators, research studies, and other relevant sources. Use these tools to gather the information needed to answer the question.

## Research Approach
- Use your tools iteratively to gather comprehensive information
- Build each query based on findings from previous searches - reference specific data points, figures, or details you've discovered to make subsequent queries more targeted
- You may use up to ${maxToolCalls} tool calls to thoroughly research the question
- Provide your final answer once you have sufficient information to respond accurately

## Final Answer
Provide a clear, complete answer that:
- Directly addresses the question asked
- Includes relevant data, numbers, and dates
- Cites sources when possible
- Is based on the information gathered through your research`;

// ============================================================================
// BENCHMARK QUERY FUNCTION
// ============================================================================

export async function queryForBenchmark(
  question: string,
  questionType: string = 'financial',
  toolChoice: string = 'valyu'
): Promise<string> {
  try {
    // Check for required Gemini API key
    if (!GOOGLE_GENERATIVE_AI_API_KEY) {
      throw new Error('GOOGLE_GENERATIVE_AI_API_KEY environment variable is not set. Please get your API key from https://aistudio.google.com/app/apikey');
    }

    const currentDate = new Date().toISOString().split('T')[0];

    console.error(`[AgentSearch] Processing question: ${question.substring(0, 100)}...`);
    console.error(`[AgentSearch] Question type: ${questionType}`);
    console.error(`[AgentSearch] Tool choice: ${toolChoice}`);

    // Select the appropriate tool based on toolChoice parameter
    let selectedTool: any;
    let toolName: string;

    switch (toolChoice.toLowerCase()) {
      case 'google':
      case 'serpapi':
        selectedTool = googleSearchTool;
        toolName = 'googleSearch';
        console.error('[AgentSearch] Using Google Search (SerpAPI)');
        break;
      case 'exa':
        selectedTool = exaSearchTool;
        toolName = 'exaSearch';
        console.error('[AgentSearch] Using Exa Search');
        break;
      case 'parallel':
        selectedTool = parallelSearchTool;
        toolName = 'parallelSearch';
        console.error('[AgentSearch] Using Parallel Search');
        break;
      case 'valyu':
      default:
        selectedTool = tool({
          description: valyuDeepSearchTool.description,
          inputSchema: z.object({
            query: z.string().describe('Detailed search query'),
          }),
          execute: async ({ query }: { query: string }) => {
            return await (valyuDeepSearchTool.execute as any)({ query });
          },
        });
        toolName = 'valyuSearch';
        console.error('[AgentSearch] Using Valyu Search');
        break;
    }

    // Select the appropriate system prompt based on question type
    let systemPrompt: string;
    switch (questionType.toLowerCase()) {
      case 'medical':
        systemPrompt = createMedicalSystemPrompt(question, currentDate, MAX_TOOL_CALLS);
        console.error('[AgentSearch] Using Medical system prompt');
        break;
      case 'economics':
      case 'economic':
        systemPrompt = createEconomicSystemPrompt(question, currentDate, MAX_TOOL_CALLS);
        console.error('[AgentSearch] Using Economics system prompt');
        break;
      case 'financial':
      case 'finance':
      default:
        systemPrompt = createFinanceSystemPrompt(question, currentDate, MAX_TOOL_CALLS);
        console.error('[AgentSearch] Using Finance system prompt');
        break;
    }

    // Generate response using Gemini with selected tool
    const { text, response } = await generateText({
      model: google(GEMINI_MODEL),
      tools: {
        [toolName]: selectedTool,
      },
      prompt: systemPrompt,
      stopWhen: stepCountIs(MAX_TOOL_CALLS + 1),
    });

    // Extract tool outputs from response messages
    const toolOutputs: any[] = [];

    if (response.messages && response.messages.length > 0) {
      for (const message of response.messages) {
        if (message.role === 'tool') {
          try {
            const messageContent = Array.isArray(message.content) ? message.content[0] : message.content;
            const output = (messageContent as any)?.output;

            if (output) {
              toolOutputs.push({
                toolName: (messageContent as any)?.toolName || 'unknown',
                toolCallId: (messageContent as any)?.toolCallId,
                output: output.value || output,
              });
            }
          } catch (error) {
            console.error('[AgentSearch] Error extracting tool output:', error);
          }
        }
      }
    }

    const benchmarkResult = {
      text: text,
      toolOutputs: toolOutputs,
      messageCount: response.messages.length
    };

    console.error(`[AgentSearch] Query completed successfully`);

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
      console.error('Usage: node query.ts --benchmark --question "Your question here" [--type financial] [--tool valyu]');
      process.exit(1);
    }

    const question = args[questionIndex + 1];
    const questionType = typeIndex !== -1 && typeIndex + 1 < args.length ? args[typeIndex + 1] : 'financial';
    const toolChoice = toolIndex !== -1 && toolIndex + 1 < args.length ? args[toolIndex + 1] : 'valyu';

    try {
      console.error(`[AgentSearch] Starting benchmark query...`);
      const response = await queryForBenchmark(question, questionType, toolChoice);
      console.log(response);
    } catch (error) {
      console.error(`[AgentSearch] Error: ${error}`);
      process.exit(1);
    }
  } else {
    console.error('Error: --benchmark flag is required');
    console.error('Usage: node query.ts --benchmark --question "Your question here" [--type financial] [--tool valyu]');
    process.exit(1);
  }
}

// Run the script
const isMainModule = process.argv[1] && process.argv[1].endsWith('query.ts');
if (isMainModule) {
  main().catch(console.error);
}
