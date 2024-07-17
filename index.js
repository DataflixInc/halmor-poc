// import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";

import * as fs from "fs";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import dotenv from "dotenv";

import { WatsonXAIEmbeddings } from "./Watsonxai.embeddings.js";

dotenv.config();

const generateEmbeddings = async () => {
    try {
        // Path to Knowledge Base
        const DATA_FILE_PATH = "./data/data.txt";
        const data = fs.readFileSync(DATA_FILE_PATH, "utf8");

        // Splitting the data 
        const textSplitterChat = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 150,
        });

        // Doc is like [{pageContent: string, metadata: object}]
        const docs = await textSplitterChat.createDocuments([data]);

        // Generate embeddings using Watsonx.ai
        const vectorStore = await HNSWLib.fromDocuments(
            docs,
            new WatsonXAIEmbeddings({})
        );

        // save embeddings into the embeddings folder of the RAG
        await vectorStore.save("./rag/src/embeddings");
        await vectorStore.save("./rag/bin/embeddings");
        return;
    } catch (error) {
        console.log(error);
    }
};

await generateEmbeddings();
  