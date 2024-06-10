import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "@langchain/openai";
import { WatsonxAI } from "@langchain/community/llms/watsonx_ai";
import { formatDocumentsAsString } from "langchain/util/document";
import { PromptTemplate } from "@langchain/core/prompts";
import {
    RunnableSequence,
    RunnablePassthrough,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

import {
    ChatPromptTemplate,
    MessagesPlaceholder,
} from "@langchain/core/prompts";
import {
    AIMessage,
    AIMessageFields,
    BaseMessageFields,
    HumanMessage,
} from "@langchain/core/messages";
import { response } from "express";

interface ChatItem {
    a?: string;
    u?: string;
}

type Message = AIMessage | HumanMessage | undefined; // Common interface for both message types

export const generate = async (chatHistory: ChatItem[]) => {
    try {
        console.log("Generating");
        const question = chatHistory.splice(chatHistory.length - 1, 1)[0]["u"];
        console.log("Question", question);
        const chat = formatChatHistory(chatHistory);
        console.log("Chat", chat);
        const vectorStoreRetriever = await createVectorStoreRetriever();

        const model = new WatsonxAI({
            modelId: "meta-llama/llama-3-8b-instruct",
            modelParameters: {
                max_new_tokens: 250,
                temperature: 0.5,
                stop_sequences: [],
                repetition_penalty: 1,
            },
        });

        const contextualQ =
            chat.length <= 2
                ? await contextualizedQ(chat, question!)
                : question!;

        console.log("Contextual Q", contextualQ);

        const response = await finalChain(
            model,
            vectorStoreRetriever,
            chat,
            contextualQ
        );
        console.log("Response", response);

        let cleanedResponse = cleanResponse(response);
        console.log("Cleaned Response", cleanedResponse);

        return cleanedResponse;
    } catch (error) {
        console.log(error);
    }
};

const createVectorStoreRetriever = async () => {
    console.log("Creating Vector Store Retriever");
    // Load the vector store
    const vectorStore = await HNSWLib.load(
        __dirname + "/embeddings",
        new OpenAIEmbeddings()
    );
    return vectorStore.asRetriever();
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

const finalChain = async (
    model: WatsonxAI,
    retriever: any,
    chatHistory: Message[],
    question: string
) => {
    try {
        // You are a knowledgeable nutritionist.
        // Your goal is to provide personalized helpful, accurate and concise answers to nutrition related questions specifically keto.
        // Your responses should be engaging, and informative, making users feel like they are chatting with a knowledgeable nutritionist.
        // Keep your responses short, positive and easy to understand.
        // Your responses should be engaging, and informative, making users feel like they are chatting with a knowledgeable human coach.

        // Use the context from the provided dataset of YouTube transcripts to enhance your answers with relevant examples, tips, and explanations.
        // The context is part of conversations in a youtube channel.

        // -----
        // You are a knowledgeable nutritionist.
        // Given a chat history and the user message, return response should help user achieve their goals.
        // Use the context from the provided dataset of YouTube transcripts to enhance your answers with relevant examples, tips, and explanations.
        // If the user interaction is a question and directly pertains to the context of the uploaded document, provide an answer to the question.
        // Please ensure that the responses are accurate and contextually relevant to the document content.
        // Phrase the response as your own personal opinion.

        // Context: {context}
        // ----
        console.log("Final Chain");

        const qaSystemPrompt = `
        You are an AI nutrition coach focused on low carb diet and keto diet. Sound like a human.
        Given a chat history and the user message, return response should help user achieve their goals.
        Follow below instructions while generating response.
        1. Avoid Hallucination: Prohibit the inclusion of any hallucinatory or speculative content in the generated text.
        2. If human asks anything other than keto diet, diet, nutrition, lifestyle or fitness, you can respond "I don't know".
        3. Do NOT add any prefix to the response.
        4. Remove new lines and extra spaces.
        5. Use three sentences maximum and keep the response concise.
        6. Respond should be a reply to user message. Do Not include anything else other than response to user message.
        7. Phrase the response as your own personal opinion.
        8. Response has to be plain text.

        Context: {context}

        `;

        const qaPrompt = ChatPromptTemplate.fromMessages([
            ["system", qaSystemPrompt],
            new MessagesPlaceholder("chat_history"),
            ["human", "{question}"],
        ]);

        const ragChain = RunnableSequence.from([
            {
                context: async (input) => {
                    const relevantDocs = await retriever.getRelevantDocuments(
                        input.question
                    );
                    console.log("Relevant Docs", relevantDocs);
                    return formatDocumentsAsString(relevantDocs);
                },
                chat_history: (input) => input.chat_history,
                question: (input) => input.question,
            },
            qaPrompt,
            model,
        ]);

        return await ragChain.invoke({
            chat_history: chatHistory,
            question,
        });
    } catch (error) {
        console.log("Error in final chain", error);
        return "Error in final chain";
    }
};

const formatChatHistory = (chatHistory: ChatItem[]): Message[] => {
    console.log("Formatting Chat History");
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
    let res = response.replace(/\s*AI:\s*/gm, "");
    res = res.replace(/(\r\n|\n|\r)/gm, "");
    return res;
};
