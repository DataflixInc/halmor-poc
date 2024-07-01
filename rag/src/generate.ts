import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { WatsonxAI } from "@langchain/community/llms/watsonx_ai";
import { formatDocumentsAsString } from "langchain/util/document";

import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

import {
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
} from "@langchain/core/prompts";
import { AIMessage, HumanMessage } from "@langchain/core/messages";

interface ChatItem {
    a?: string;
    u?: string;
}

import { WatsonXAIEmbeddings } from "./Watsonxai.embeddings";

type Message = AIMessage | HumanMessage | undefined; // Common interface for both message types

export const generate = async (chatHistory: ChatItem[]) => {
    try {
        const question = chatHistory.splice(chatHistory.length - 1, 1)[0]["u"];
        const chat = formatChatHistory(chatHistory);
        const docs = await vectorStoreDocs(question as string);

        const model = new WatsonxAI({
            modelId: "meta-llama/llama-3-70b-instruct",
            modelParameters: {
                max_new_tokens: 500,
                temperature: 0.3,
                stop_sequences: [],
                repetition_penalty: 1,
            },
        });

        const contextualQ =
            chat.length >= 2
                ? await contextualizedQ(chat, question!)
                : question!;

        const response = await finalChain(model, docs, contextualQ);
        console.log("Response", response);

        let cleanedResponse = cleanResponse(response);
        console.log("Cleaned Response", cleanedResponse);

        return cleanedResponse;
    } catch (error) {
        console.log(error);
    }
};

const vectorStoreDocs = async (question: string) => {
    console.log("Creating Vector Store Retriever");
    // Load the vector store
    const vectorStore = await HNSWLib.load(
        __dirname + "/embeddings",
        new WatsonXAIEmbeddings({})
    );
    return await vectorStore.similaritySearch(question, 1);
};

const contextualizedQ = async (
    chat: Message[],
    question: string
): Promise<string> => {
    try {
        console.log("Contextualized Question");

        const model = new WatsonxAI({
            modelId: "google/flan-t5-xl",
            modelParameters: {
                max_new_tokens: 200,
                temperature: 0,
                stop_sequences: [],
                repetition_penalty: 1,
            },
        });

        const contextualizeQSystemPrompt = `
        Given a chat history and the user message which might reference context in the chat history, rewrite the users message which can be understood without the chat history.
        Strictly follow below instructions while giving the answer:
        1. Do NOT respond to the user message.
        2. Only rewrite the message if needed, otherwise return the message as is.
        `;

        const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
            ["system", contextualizeQSystemPrompt],
            new MessagesPlaceholder("chat_history"),
            ["human", "message: {question}"],
        ]);
        const contextualizeQChain = contextualizeQPrompt
            .pipe(model)
            .pipe(new StringOutputParser());

        return await contextualizeQChain.invoke({
            chat_history: chat,
            question: question,
        });
    } catch (error) {
        console.log("Error contextualizing question", error);
        return question;
    }
};

const finalChain = async (model: WatsonxAI, docs: any, question: string) => {
    try {
        console.log("Final Chain");

        const qaSystemPrompt = PromptTemplate.fromTemplate(`
        You are pro KetoCoach, an AI nutrition coach with expertise in low-carb and keto diets.
        You provide personalized advice, meal plans, and tips to help users successfully follow these dietary plans.
        Your responses should be engaging, and informative, making users feel like they are chatting with a knowledgeable human coach.
        Use the context from the provided dataset of YouTube transcripts to enhance your answers with relevant examples, tips, and explanations.
        Phrase the response as your own personal opinion.
        Only use multiple sentences when it’s necessary to convey the meaning of your response in longer responses. 
        You can use only a maximum of 3 sentences or 3 items.
        Respond only to the question. 

        Context: {context}

        Question: {question}

        Response:
        `);

        const ragChain = RunnableSequence.from([
            {
                context: async (input) => {
                    const relevantDocs = docs;
                    console.log("Relevant Docs", relevantDocs);
                    return formatDocumentsAsString(relevantDocs);
                },
                question: (input) => input.question,
            },
            qaSystemPrompt,
            model,
        ]);

        return await ragChain.invoke({
            question,
        });
    } catch (error) {
        console.log("Error in final chain", error);
        return "Error in final chain";
    }
};

const formatChatHistory = (chatHistory: ChatItem[]): Message[] => {
    let newChatHistory: ChatItem[] = [];
    if (chatHistory.length > 6)
        newChatHistory = chatHistory.slice(
            chatHistory.length - 6,
            chatHistory.length
        );
    else newChatHistory = chatHistory;

    return chatHistory.map((item: ChatItem) => {
        if (item.a) {
            return new AIMessage(item.a);
        } else if (item.u) {
            return new HumanMessage(item.u);
        }
    });
};

const cleanResponse = (response: string) => {
    console.log("Cleaning Response");
    let res = response.replace(/\s\s+/g, " ");
    res = res.replace(/\s*AI:\s*/gm, "");
    res = res.replace(/(\r\n|\n|\r)/gm, "");
    res = res.trim();
    return res;
};
