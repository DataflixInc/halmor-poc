import { WatsonXAI } from "@ibm-cloud/watsonx-ai";
import { IamAuthenticator } from "ibm-watson/auth/index.js";
import { Embeddings } from "@langchain/core/embeddings";

export class WatsonXAIEmbeddings extends Embeddings {
    config;
    constructor(config) {
        super(config);
        this.config = config;
    }

    async embedDocuments(documents) {
        try {
            let watsonxAIService = WatsonXAI.newInstance({
                version: "2024-05-31",
                serviceUrl: "https://us-south.ml.cloud.ibm.com",
                authenticator: new IamAuthenticator({
                    apikey: process.env.IBM_CLOUD_API_KEY,
                }),
            });

            const processedDocuments = processDocs(documents);

            const generatedEmbeddings = await Promise.all(
                processedDocuments.map(async (chunk) => {
                    const embeddingParameters = {
                        inputs: chunk,
                        modelId: "ibm/slate-125m-english-rtrvr",
                        projectId: process.env.WATSONX_PROJECT_ID,
                    };

                    const embeddings = await watsonxAIService.textEmbeddings(
                        embeddingParameters
                    );
                    return embeddings;
                })
            );

            const finalEmbeddings = generatedEmbeddings.map((embeddings) => {
                console.log("Embeddings", embeddings.result.results);
                return embeddings.result.results;
            });

            let embeddings = [];
            for (let i = 0; i < finalEmbeddings.length; i++) {
                for (let j = 0; j < finalEmbeddings[i].length; j += 1) {
                    embeddings.push(finalEmbeddings[i][j].embedding);
                }
            }

            return embeddings;
        } catch (error) {
            console.log("error in embedding", error);
            throw error;
        }
    }

    async embedQuery(query) {
        // Service instance
        let watsonxAIService = WatsonXAI.newInstance({
            version: "2024-05-31",
            serviceUrl: "https://us-south.ml.cloud.ibm.com",
            authenticator: new IamAuthenticator({
                apikey: process.env.IBM_CLOUD_API_KEY,
            }),
        });

        const embeddingParameters = {
            inputs: [query],
            modelId: "ibm/slate-125m-english-rtrvr",
            projectId: process.env.WATSONX_PROJECT_ID,
        };

        const generatedEmbeddings = await watsonxAIService.textEmbeddings(
            embeddingParameters
        );

        const finalEmbeddings = generatedEmbeddings.result.results.map(
            (embedding) => embedding
        );
        console.log("Final Embeddings", finalEmbeddings);

        return finalEmbeddings;
    }
    catch(error) {
        console.log("error in query embedding", error);
        throw error;
    }
}

const processDocs = (docs) => {
    const chunkSize = 1000;
    const finalChunks = [];
    for (let i = 0; i < docs.length; i += chunkSize) {
        const chunk = docs.slice(i, i + chunkSize);
        finalChunks.push(chunk);
    }
    return finalChunks;
};
