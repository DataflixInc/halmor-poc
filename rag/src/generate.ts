import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { WatsonxAI } from "@langchain/community/llms/watsonx_ai";
import { formatDocumentsAsString } from "langchain/util/document";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RunnableSequence } from "@langchain/core/runnables";

import {
    ChatPromptTemplate,
} from "@langchain/core/prompts";


export const generate = async (question: string) => {
    try {
        const response = await finalChain(question!);
        console.log("Response", response);

        // let cleanedResponse = cleanResponse(response);
        // console.log("Cleaned Response", cleanedResponse);

        return response;
    } catch (error) {
        console.log(error);
    }
};

const vectorStoreRetriever = async (question: string) => {
    // Load the vector store
    const vectorStore = await HNSWLib.load(
        __dirname + "/embeddings",
        // new WatsonXAIEmbeddings({})
        new OpenAIEmbeddings({
            openAIApiKey: process.env.OPENAI_API_KEY,
        })
    );
    return await vectorStore.similaritySearch(question, 5)
};


const finalChain = async (question: string) => {
    try {
        console.log("Final Chain");

        const model = new WatsonxAI({
            modelId: "meta-llama/llama-3-70b-instruct",
            modelParameters: {
                max_new_tokens: 1000,
                temperature: 0.7,
                stop_sequences: ["Human:"],
                repetition_penalty: 1,
            },
        });

        const qaSystemPrompt = `
        Never use the terms such as 'text or context 1, 2 or 3' in your answer as the questioner may not know how the texts are retrieved. 
        If the question is in English, answer in English. 
        If the question is in Spanish, answer in Spanish and similarly if the question is in XYZ language, answer it in the same XYZ language. 
        If you do not know the answer, you can say 'I do not have information about that'. 
        Answer in detail. 

        Context: {context}
        `;

        const qaPrompt = ChatPromptTemplate.fromMessages([
            ["system", qaSystemPrompt],
            ["human", "{question}"],
            ["ai", "Response:"],
        ]);


        const ragChain = RunnableSequence.from([
            {
                context: async (input: Record<string, unknown>) => {
                    const retrievedDocs = await vectorStoreRetriever(question);
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
        });
    } catch (error) {
        console.log("Error in final chain", error);
        return "Error in final chain";
    }
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
