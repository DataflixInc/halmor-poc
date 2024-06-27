// import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";

import * as fs from "fs";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import dotenv from "dotenv";
import { OpenAIEmbeddings } from "@langchain/openai";
// import { WatsonXAIEmbeddings } from "./Watsonxai.embeddings.js";

dotenv.config();

const generateEmbeddings = async () => {
    try {
        // const embeddingsModel = new OpenAIEmbeddings({
        //     openAIApiKey: "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",
        //     maxConcurrency: 5,
        // });
        console.log("Generating embeddings.");
        // const embeddingsModel = new HuggingFaceTransformersEmbeddings({
        //     model: "Xenova/all-MiniLM-L6-v2",
        // });

        // Text data path
        const DATA_FILE_PATH = "./data/data.txt";
        const data = fs.readFileSync(DATA_FILE_PATH, "utf8");

        const textSplitterChat = new RecursiveCharacterTextSplitter({
            chunkSize: 200,
            chunkOverlap: 0,
        });

        const docs = await textSplitterChat.createDocuments([data]);

        // const finalEmbeddings = await generateEmbeddingsWithWatsonx(docs);

        const vectorStore = await HNSWLib.fromDocuments(
            docs,
            // new WatsonXAIEmbeddings()
            new OpenAIEmbeddings()
        );
        await vectorStore.save("./rag/src/embeddings");
        await vectorStore.save("./rag/bin/embeddings");
        return;
    } catch (error) {
        console.log(error);
    }
};

await generateEmbeddings();

// {
//     object: 'list',
//     data: [
//       { object: 'embedding', index: 0, embedding: [Array] },
//       { object: 'embedding', index: 0, embedding: [Array] },
//       { object: 'embedding', index: 0, embedding: [Array] },
//     ],
//     model: 'text-embedding-ada-002',
//     usage: { prompt_tokens: 23065, total_tokens: 23065 }
//   }

/*
Open AI Embeddings
    {
  object: 'embedding',
  index: 0,
  embedding: [
       -0.01089892,   -0.009014992,   0.024013443,    0.006043162,
       0.007880655,    0.038952194,  -0.012723145,   -0.060975574,
      0.0032471218,   0.0014776228,   0.007336704,    0.028842667,
      -0.008285302,   -0.016504267,  -0.008305202,   -0.020988546,
       0.029001871,    0.009492607,   0.014713209,   -0.031045005,
      -0.015588838,   -0.005694901, -0.0016227317,  -0.0013648525,
       0.017207423,   -0.011323466, -0.0046335333,    -0.02092221,
      -0.018268792,    0.014036587,   0.016862478,   -0.014487669,
       0.023164349,    -0.01503162, -0.0035456314,   -0.004461061,
       0.006633548,   0.0008428752,   0.011781181,   -0.007973525,
       0.009393104,   -0.015204092,   0.018427996,   -0.013996786,
       -0.02727715,  -0.0025124564,  -0.016159322,   -0.014978551,
     -0.0005493407,    0.007993425,   0.032053303,  -0.0018988531,
      -0.018746406,   -0.023734834,  0.0048259064,    0.024942141,
      0.0021409777,    0.012000089,  0.0145805385,   0.0071310643,
       0.012802748,    0.029824432, -0.0022437975,    0.022182584,
      0.0107927825,   0.0043482906,  -0.022169318,    0.029983636,
        0.02335009,    0.008616979,   0.025565693,    0.015044887,
     -0.0064179576,    0.012338399,   0.043648746,    -0.02828545,
    -0.00022118737,   -0.014912216, -0.0040564146,      0.0195955,
            0.0227,   -0.021864174,   -0.01727376,    0.023230685,
        0.01525716, -0.00090299174,  0.0046899184,    0.009976856,
      0.0059967274, -0.00088640786,   0.026096378,  0.00013940816,
      0.0026152763,    0.030912334,  -0.040225834, -0.00038972095,
      -0.021161018,    0.021452894,  -0.021068148,    0.003432861,
    ... 1436 more items
  ]
}

*/
