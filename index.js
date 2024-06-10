// import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { OpenAIEmbeddings } from "@langchain/openai";

import * as fs from "fs";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const generateEmbeddings = async () => {
    try {
        const embeddingsModel = new OpenAIEmbeddings({
            openAIApiKey: "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",
            maxConcurrency: 5,
        });
        console.log("Generating embeddings.");
        // const embeddingsModel = new HuggingFaceTransformersEmbeddings({
        //     model: "Xenova/all-MiniLM-L6-v2",
        // });

        // Text data path
        const DATA_FILE_PATH = "./data/data.txt";
        const data = fs.readFileSync(DATA_FILE_PATH, "utf8");
        console.log(data);

        const textSplitterChat = new RecursiveCharacterTextSplitter({
            chunkSize: 2000,
            chunkOverlap: 500,
        });

        const docs = await textSplitterChat.createDocuments([data]);

        const vectorStore = await HNSWLib.fromDocuments(docs, embeddingsModel);
        await vectorStore.save("./rag/src/embeddings");
        await vectorStore.save("./rag/bin/embeddings");
        return;
    } catch (error) {
        console.log(error);
    }
};

await generateEmbeddings();
