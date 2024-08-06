// import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";

import * as fs from "fs";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import dotenv from "dotenv";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { OpenAIEmbeddings } from "@langchain/openai";

dotenv.config();

const generateEmbeddings = async () => {
    try {
        const pdfPath = "./data/pdfs/";
        const directoryLoader = new DirectoryLoader(pdfPath, {
            ".pdf": (path) => new PDFLoader(path),
          });

        const directoryDocs = await directoryLoader.load();

        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 2000,
            chunkOverlap: 250,
        });
        
        const splitDocs = await textSplitter.splitDocuments(directoryDocs);

        // Generate embeddings using Watsonx.ai
        console.log("Generating Embeddings...")
        const vectorStore = await HNSWLib.fromDocuments(
            splitDocs,
            new OpenAIEmbeddings({
                openAIApiKey: process.env.OPENAI_API_KEY,
            })
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
  