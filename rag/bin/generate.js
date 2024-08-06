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
exports.generate = void 0;
const hnswlib_1 = require("@langchain/community/vectorstores/hnswlib");
const watsonx_ai_1 = require("@langchain/community/llms/watsonx_ai");
const document_1 = require("langchain/util/document");
const runnables_1 = require("@langchain/core/runnables");
const prompts_1 = require("@langchain/core/prompts");
const Watsonxai_embeddings_1 = require("./Watsonxai.embeddings");
const generate = (question) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const response = yield finalChain(question);
        console.log("Response", response);
        let cleanedResponse = cleanResponse(response);
        console.log("Cleaned Response", cleanedResponse);
        return cleanedResponse;
    }
    catch (error) {
        console.log(error);
    }
});
exports.generate = generate;
const vectorStoreRetriever = (question) => __awaiter(void 0, void 0, void 0, function* () {
    // Load the vector store
    const vectorStore = yield hnswlib_1.HNSWLib.load(__dirname + "/embeddings", new Watsonxai_embeddings_1.WatsonXAIEmbeddings({}));
    return yield vectorStore.similaritySearch(question, 10);
});
const finalChain = (question) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        console.log("Final Chain");
        const model = new watsonx_ai_1.WatsonxAI({
            modelId: "meta-llama/llama-3-70b-instruct",
            modelParameters: {
                max_new_tokens: 500,
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
        const qaPrompt = prompts_1.ChatPromptTemplate.fromMessages([
            ["system", qaSystemPrompt],
            ["human", "{question}"],
            ["ai", "Response:"],
        ]);
        const ragChain = runnables_1.RunnableSequence.from([
            {
                context: (input) => __awaiter(void 0, void 0, void 0, function* () {
                    const retrievedDocs = yield vectorStoreRetriever(question);
                    console.log("Retrieved Docs", retrievedDocs);
                    return (0, document_1.formatDocumentsAsString)(retrievedDocs);
                }),
                chat_history: (input) => input.chat,
                question: (input) => input.question,
            },
            qaPrompt,
            model,
        ]);
        return yield ragChain.invoke({
            question: question,
        });
    }
    catch (error) {
        console.log("Error in final chain", error);
        return "Error in final chain";
    }
});
const cleanResponse = (response) => {
    console.log("Cleaning Response");
    let res = response.replace(/\s\s+/g, " ");
    res = res.replace(/\s*AI:\s*/gm, "");
    res = res.replace(/(\r\n|\n|\r)/gm, "");
    res = res.trim();
    if (res.includes("Human:")) {
        // Replace everything after Human: with nothing
        res = res.replace(/Human:.*/gm, "");
    }
    return res;
};
//# sourceMappingURL=generate.js.map