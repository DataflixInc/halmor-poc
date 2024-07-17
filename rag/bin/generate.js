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
const output_parsers_1 = require("@langchain/core/output_parsers");
const prompts_1 = require("@langchain/core/prompts");
const messages_1 = require("@langchain/core/messages");
const Watsonxai_embeddings_1 = require("./Watsonxai.embeddings");
const generate = (chatHistory) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const question = chatHistory.splice(chatHistory.length - 1, 1)[0]["u"];
        console.log("Question", question);
        console.log("Chat History Length", chatHistory.length);
        const strippedChat = chatHistory.length >= 5 ? chatHistory.splice(chatHistory.length - 5, chatHistory.length) : chatHistory;
        console.log("Stripped Chat", strippedChat);
        const chat = formatChatHistory(strippedChat);
        console.log("Formatted Chat", chat);
        const response = yield finalChain(chat, question);
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
    return yield vectorStore.similaritySearch(question, 5);
});
const contextualQChain = (chatHistory, question) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        if (chatHistory.length === 1) {
            console.log("No Chat History");
            return question;
        }
        else {
            const model = new watsonx_ai_1.WatsonxAI({
                modelId: "mistralai/mistral-large",
                modelParameters: {
                    max_new_tokens: 200,
                    temperature: 0.5,
                    stop_sequences: [],
                    repetition_penalty: 1,
                },
            });
            const contextualizeQSystemPrompt = `
                Rewrite the users question which can be understood without the chat history.
                ONLY Rephrase and return the question in a way that is has context from the chat history.
            `;
            const contextualizeQPrompt = prompts_1.ChatPromptTemplate.fromMessages([
                ["system", contextualizeQSystemPrompt],
                new prompts_1.MessagesPlaceholder("chat_history"),
                ["human", "{question}"],
                ["ai", "Rephrased Question:"],
            ]);
            const contextualizeQChain = contextualizeQPrompt
                .pipe(model)
                .pipe(new output_parsers_1.StringOutputParser());
            const res = yield contextualizeQChain.invoke({
                chat_history: chatHistory,
                question: question
            });
            if (res == "") {
                console.log("No Contextualized Q");
                return question;
            }
            console.log("Contextualized Q", res);
            return res;
        }
    }
    catch (error) {
        console.log("Error contextualizing question", error);
        throw error;
    }
});
const finalChain = (chat, question) => __awaiter(void 0, void 0, void 0, function* () {
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
        You are pro KetoCoach, an AI nutrition coach with expertise in low-carb and keto diets.
        You provide personalized advice, meal plans, and tips to help users successfully follow these dietary plans.
        Your responses should be engaging, and informative, making users feel like they are chatting with a knowledgeable human coach.
        Use the provided context enhance your answers with relevant examples, tips, and explanations.
        Your answers have to be informational. Do not answer in your perspective.
        Only use multiple sentences when it’s necessary to convey the meaning of your response in longer responses. 
        You can use only a maximum of 3 sentences or 3 items.
        Respond only to the question. Respond with your answer. Do not complete conversation for the user.

        Context: {context}
        `;
        const qaPrompt = prompts_1.ChatPromptTemplate.fromMessages([
            ["system", qaSystemPrompt],
            new prompts_1.MessagesPlaceholder("chat_history"),
            ["human", "{question}"],
            ["ai", "Response:"],
        ]);
        const contextualizeQChainRes = yield contextualQChain(chat, question);
        const ragChain = runnables_1.RunnableSequence.from([
            {
                context: (input) => __awaiter(void 0, void 0, void 0, function* () {
                    const retrievedDocs = yield vectorStoreRetriever(contextualizeQChainRes);
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
            chat: chat,
        });
    }
    catch (error) {
        console.log("Error in final chain", error);
        return "Error in final chain";
    }
});
const formatChatHistory = (chatHistory) => {
    return chatHistory.map((item) => {
        if (item.a) {
            return new messages_1.AIMessage(item.a);
        }
        else if (item.u) {
            return new messages_1.HumanMessage(item.u);
        }
    });
};
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