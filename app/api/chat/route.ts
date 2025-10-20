import { NextRequest, NextResponse } from 'next/server';
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ChatGroq } from "@langchain/groq";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { TavilySearch } from "@langchain/tavily";
import { MemorySaver } from '@langchain/langgraph';
import { WikipediaQueryRun } from "@langchain/community/tools/wikipedia_query_run";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";

// Initialize checkpointer
const checkpointer = new MemorySaver();

// Initialize Tavily tool
const tavilyTool = new TavilySearch({
    maxResults: 3,
    apiKey: process.env.TAVILY_API_KEY, 
    topic: 'general',
    includeAnswer: true,
    searchDepth: "basic",
});

// Initialize Wikipedia tool
const wikiTool = new WikipediaQueryRun({
    topKResults: 2,
    maxDocContentLength: 2000,
});

const loggingWikiTool = new DynamicStructuredTool({
    name: "wikipedia",
    description: "Search Wikipedia for encyclopedic information about a topic. Use this for factual, historical, or general knowledge questions.",
    schema: z.object({
        query: z.string().describe("The topic to search on Wikipedia"),
    }),
    func: async ({ query }) => {
        console.log("📚 Calling Wikipedia tool...");
        return await wikiTool.invoke(query);
    },
});

const loggingNewsTool = new DynamicStructuredTool({
    name: "news_search",
    description: "Search for recent news articles on a specific topic. Use this when the user asks for news, current events, or recent updates.",
    schema: z.object({
        query: z.string().describe("The news topic or query to search for"),
    }),
    func: async ({ query }) => {
        console.log("📰 Calling News tool...");
        const newsSearch = new TavilySearch({
            maxResults: 3,
            apiKey: process.env.TAVILY_API_KEY,
            topic: 'news',
            includeAnswer: true,
            searchDepth: "basic",
        });
        return await newsSearch.invoke(query);
    },
});

// Add all tools to the array
const tools = [tavilyTool, loggingWikiTool, loggingNewsTool];
const toolNode = new ToolNode(tools);

// Initialize LLM 
const llm = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0,
  maxRetries: 2,
  apiKey: process.env.GROQ_API_KEY,
}).bindTools(tools);

// Call LLM
async function callModel(state: any) {
  const response = await llm.invoke(state.messages);
  return { messages: [response] };
}

// Conditional edge
function shouldContinue(state: any) {
    const lastMessage = state.messages[state.messages.length - 1];
    
    if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
        return 'tools';
    }
    return '__end__';
}

// Define workflow
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge("__start__", "agent")
  .addConditionalEdges('agent', shouldContinue, {
    tools: "tools",
    __end__: "__end__"
  })
  .addEdge('tools','agent');

const app = workflow.compile({ checkpointer });

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { message, threadId } = body;

    if (!message) {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      );
    }

    console.log('🔄 Processing message:', message);

    // Invoke the LangGraph agent
    const finalState = await app.invoke(
      { messages: [{ role: "user", content: message }] },
      { configurable: { thread_id: threadId || "default-thread" } }
    );

    const lastMessage = finalState.messages[finalState.messages.length - 1];
    
    console.log('✅ Response generated:', lastMessage.content);

    return NextResponse.json({
      response: lastMessage.content,
      success: true
    });

  } catch (error: any) {
    console.error('❌ Error in chat API:', error);
    return NextResponse.json(
      { 
        error: 'Failed to process message',
        details: error.message 
      },
      { status: 500 }
    );
  }
}