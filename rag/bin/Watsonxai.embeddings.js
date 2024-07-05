"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.WatsonXAIEmbeddings = void 0;
const watsonx_ai_1 = require("@ibm-cloud/watsonx-ai");
const index_js_1 = require("ibm-watson/auth/index.js");
const embeddings_1 = require("@langchain/core/embeddings");
class WatsonXAIEmbeddings extends embeddings_1.Embeddings {
    constructor(config) {
        super(config);
        this.config = config;
    }
    embedDocuments(documents) {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                let watsonxAIService = watsonx_ai_1.WatsonXAI.newInstance({
                    version: "2024-05-31",
                    serviceUrl: "https://us-south.ml.cloud.ibm.com",
                    authenticator: new index_js_1.IamAuthenticator({
                        apikey: process.env.IBM_CLOUD_API_KEY,
                    }),
                });
                const processedDocuments = processDocs(documents);
                const generatedEmbeddings = yield Promise.all(processedDocuments.map((chunk) => __awaiter(this, void 0, void 0, function* () {
                    const embeddingParameters = {
                        inputs: chunk,
                        modelId: "ibm/slate-125m-english-rtrvr",
                        projectId: process.env.WATSONX_PROJECT_ID,
                    };
                    const embeddings = yield watsonxAIService.textEmbeddings(embeddingParameters);
                    return embeddings;
                })));
                const finalEmbeddings = generatedEmbeddings.map((embeddings) => {
                    return embeddings.result.results;
                });
                let embeddings = [];
                for (let i = 0; i < finalEmbeddings.length; i++) {
                    for (let j = 0; j < finalEmbeddings[i].length; j += 1) {
                        embeddings.push(finalEmbeddings[i][j].embedding);
                    }
                }
                return embeddings;
            }
            catch (error) {
                console.log("error in embedding", error);
                throw error;
            }
        });
    }
    embedQuery(query) {
        return __awaiter(this, void 0, void 0, function* () {
            // Service instance
            let watsonxAIService = watsonx_ai_1.WatsonXAI.newInstance({
                version: "2024-05-31",
                serviceUrl: "https://us-south.ml.cloud.ibm.com",
                authenticator: new index_js_1.IamAuthenticator({
                    apikey: process.env.IBM_CLOUD_API_KEY,
                }),
            });
            const generatedEmbeddings = yield watsonxAIService.textEmbeddings({
                inputs: [query],
                modelId: "ibm/slate-125m-english-rtrvr",
                projectId: process.env.WATSONX_PROJECT_ID,
            });
            const finalEmbeddings = generatedEmbeddings.result.results.map((embedding) => embedding);
            return finalEmbeddings[0].embedding;
        });
    }
    catch(error) {
        console.log("error in embedding", error);
        throw error;
    }
}
exports.WatsonXAIEmbeddings = WatsonXAIEmbeddings;
const processDocs = (docs) => {
    const chunkSize = 1000;
    let finalChunks = [];
    for (let i = 0; i < docs.length; i += chunkSize) {
        const chunk = docs.slice(i, i + chunkSize);
        finalChunks.push(chunk);
    }
    return finalChunks;
};
//# sourceMappingURL=Watsonxai.embeddings.js.map