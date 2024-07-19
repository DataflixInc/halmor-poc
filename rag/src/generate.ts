import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { WatsonxAI } from "@langchain/community/llms/watsonx_ai";
import { formatDocumentsAsString } from "langchain/util/document";

import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

import {
    ChatPromptTemplate,
    MessagesPlaceholder,
} from "@langchain/core/prompts";
import { AIMessage, HumanMessage } from "@langchain/core/messages";

interface ChatItem {
    a?: string;
    u?: string;
}

import { WatsonXAIEmbeddings } from "./Watsonxai.embeddings";

type Message = AIMessage | HumanMessage | undefined; // Common interface for both message types

const filterChatHistory = (chatHistory: ChatItem[]): ChatItem[] => {
    console.log("Filtering Chat History", chatHistory);
    const filteredChatHistory = chatHistory.map((item: ChatItem) => {
        if(item.a){
            // if the text contains "option: [anything]", remove it
            const optionRegex = /option:\s*\[(.*)\]/gm;
            const match = optionRegex.exec(item.a);
            if(match){
                const replacingText = item.a.replace(match[0], "");
                if(replacingText.length > 0){
                    return {
                        a: replacingText
                    };
                }
            }
            return item;
        }
        return item;
    });
    return filteredChatHistory;
};

export const generate = async (chatHistory: ChatItem[]) => {
    try {
        const question = chatHistory.splice(chatHistory.length - 1, 1)[0]["u"];
        const filteredChatHistory = filterChatHistory(chatHistory);
        console.log("Filtered Chat History", filteredChatHistory);
        const strippedChat = filteredChatHistory.length >= 5 ? filteredChatHistory.splice(filteredChatHistory.length - 5, filteredChatHistory.length) : filteredChatHistory;
        console.log("Stripped Chat", strippedChat);
        const chat = formatChatHistory(strippedChat);
        console.log("Formatted Chat", chat);

        const response = await finalChain(chat, question!);
        console.log("Response", response);

        let cleanedResponse = cleanResponse(response);
        console.log("Cleaned Response", cleanedResponse);

        return cleanedResponse;
    } catch (error) {
        console.log(error);
    }
};

const vectorStoreRetriever = async (question: string) => {
    // Load the vector store
    const vectorStore = await HNSWLib.load(
        __dirname + "/embeddings",
        new WatsonXAIEmbeddings({})
    );
    return await vectorStore.similaritySearch(question, 5);
};

const contextualQChain = async (chatHistory: Message[], question: string): Promise<string> => {
    try {

        if (chatHistory.length === 1) {
            console.log("No Chat History");
            return question;
        } else {
            const model = new WatsonxAI({
                modelId: "mistralai/mistral-large",
                modelParameters: {
                    max_new_tokens: 200,
                    temperature: 0.5,
                    stop_sequences: ["AI:", "Human:", "System:"],
                    repetition_penalty: 1,
                },
            });

            const contextualizeQSystemPrompt = `
                Rewrite the users question which can be understood without the chat history.
                Do NOT generate answer to the question.
                ONLY Rephrase question in a way that is has context from the chat history.
            `;

            const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
                ["system", contextualizeQSystemPrompt],
                new MessagesPlaceholder("chat_history"),
                ["human", "{question}"],
                ["ai", "Rephrased Question:"],
            ]);
            const contextualizeQChain = contextualizeQPrompt
                .pipe(model)
                .pipe(new StringOutputParser());

            const res = await contextualizeQChain.invoke({
                chat_history: chatHistory,
                question: question
            });

            if(res == ""){
                console.log("No Contextualized Q");
                return question;
            }
        
            console.log("Contextualized Q", res);
            return res;
        }

    } catch (error) {
        console.log("Error contextualizing question", error);
        throw error;
    }
};

const finalChain = async (chat: Message[], question: string) => {
    try {
        console.log("Final Chain");

        const model = new WatsonxAI({
            modelId: "meta-llama/llama-3-70b-instruct",
            modelParameters: {
                max_new_tokens: 500,
                temperature: 0.7,
                stop_sequences: ["Human:"],
                repetition_penalty: 1,
            },
        });

        const qaSystemPrompt = `
        You are pro KetoCoach, an AI nutrition coach with expertise in low-carb and keto diets.
        You provide personalized advice, meal plans, and tips to help users successfully follow these dietary plans.
        Your responses should be engaging, and informative, making users feel like they are chatting with a knowledgeable human coach.
        Use the provided context enhance your answers with relevant examples, tips, and explanations.
        Your answers have to be informational. Do not answer in your perspective.
        Only use multiple sentences when it’s necessary to convey the meaning of your response in longer responses. 
        You can use only a maximum of 3 sentences or 3 items.
        Respond only to the question. Respond with your answer. Do not complete conversation for the user.
        Rephrase medical terms and medical acronyms into a more general language.

        Context: {context}
        `;

        const qaPrompt = ChatPromptTemplate.fromMessages([
            ["system", qaSystemPrompt],
            new MessagesPlaceholder("chat_history"),
            ["human", "{question}"],
            ["ai", "Response:"],
        ]);

        const contextualizeQChainRes = await contextualQChain(chat, question);

        const ragChain = RunnableSequence.from([
            {
                context: async (input: Record<string, unknown>) => {
                    const retrievedDocs = await vectorStoreRetriever(contextualizeQChainRes);
                    console.log("Retrieved Docs", retrievedDocs);
                    return formatDocumentsAsString(retrievedDocs);
                },
                chat_history: (input: Record<string, unknown>) => input.chat,
                question: (input: Record<string, unknown>) => input.question,
            },
            qaPrompt,
            model,
        ]);

        return await ragChain.invoke({
            question: question,
            chat: chat,
        });
    } catch (error) {
        console.log("Error in final chain", error);
        return "Error in final chain";
    }
};

const formatChatHistory = (chatHistory: ChatItem[]): Message[] => {
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
    if(res.includes("Human:")){
        // Replace everything after Human: with nothing
        res = res.replace(/Human:.*/gm, "");
    }
    return res;
};
