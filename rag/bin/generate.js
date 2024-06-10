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
const openai_1 = require("@langchain/openai");
const watsonx_ai_1 = require("@langchain/community/llms/watsonx_ai");
const document_1 = require("langchain/util/document");
const runnables_1 = require("@langchain/core/runnables");
const output_parsers_1 = require("@langchain/core/output_parsers");
const prompts_1 = require("@langchain/core/prompts");
const messages_1 = require("@langchain/core/messages");
const generate = (chatHistory) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        console.log("Generating");
        const question = chatHistory.splice(chatHistory.length - 1, 1)[0]["u"];
        console.log("Question", question);
        const chat = formatChatHistory(chatHistory);
        console.log("Chat", chat);
        const docs = yield vectorStoreDocs(question);
        const model = new watsonx_ai_1.WatsonxAI({
            modelId: "meta-llama/llama-3-70b-instruct",
            modelParameters: {
                max_new_tokens: 500,
                temperature: 0.3,
                stop_sequences: [],
                repetition_penalty: 1,
            },
        });
        const contextualQ = chat.length <= 2
            ? yield contextualizedQ(chat, question)
            : question;
        console.log("Contextual Q", contextualQ);
        const response = yield finalChain(model, docs, contextualQ);
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
const vectorStoreDocs = (question) => __awaiter(void 0, void 0, void 0, function* () {
    console.log("Creating Vector Store Retriever");
    // Load the vector store
    const vectorStore = yield hnswlib_1.HNSWLib.load(__dirname + "/embeddings", new openai_1.OpenAIEmbeddings());
    return yield vectorStore.similaritySearch(question, 1);
});
const contextualizedQ = (chat, question) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        console.log("Contextualized Question");
        const model = new watsonx_ai_1.WatsonxAI({
            modelId: "google/flan-t5-xl",
            modelParameters: {
                max_new_tokens: 200,
                temperature: 0,
                stop_sequences: [],
                repetition_penalty: 1,
            },
        });
        const contextualizeQSystemPrompt = `
        Given a chat history and the user message which might reference context in the chat history, rewrite the users message which can be understood without the chat history.
        Strictly follow below instructions while giving the answer:
        1. Do NOT respond to the user message.
        2. Only rewrite the message if needed, otherwise return the message as is.
        `;
        const contextualizeQPrompt = prompts_1.ChatPromptTemplate.fromMessages([
            ["system", contextualizeQSystemPrompt],
            new prompts_1.MessagesPlaceholder("chat_history"),
            ["human", "message: {question}"],
        ]);
        const contextualizeQChain = contextualizeQPrompt
            .pipe(model)
            .pipe(new output_parsers_1.StringOutputParser());
        return yield contextualizeQChain.invoke({
            chat_history: chat,
            question: question,
        });
    }
    catch (error) {
        console.log("Error contextualizing question", error);
        return question;
    }
});
const finalChain = (model, docs, question) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        console.log("Final Chain");
        const qaSystemPrompt = prompts_1.PromptTemplate.fromTemplate(`
        You are pro KetoCoach, an AI nutrition coach with expertise in low-carb and keto diets.
        You provide personalized advice, meal plans, and tips to help users successfully follow these dietary plans.
        Your responses should be engaging, and informative, making users feel like they are chatting with a knowledgeable human coach.
        Use the context from the provided dataset of YouTube transcripts to enhance your answers with relevant examples, tips, and explanations.
        Phrase the response as your own personal opinion.
        Only use multiple sentences when it’s necessary to convey the meaning of your response in longer responses. 
        You can use only a maximum of 3 sentences or 3 items.
        Respond only to the question. 

        Context: {context}

        Question: {question}

        Response:
        `);
        const ragChain = runnables_1.RunnableSequence.from([
            {
                context: (input) => __awaiter(void 0, void 0, void 0, function* () {
                    const relevantDocs = docs;
                    console.log("Relevant Docs", relevantDocs);
                    return (0, document_1.formatDocumentsAsString)(relevantDocs);
                }),
                question: (input) => input.question,
            },
            qaSystemPrompt,
            model,
        ]);
        return yield ragChain.invoke({
            question,
        });
    }
    catch (error) {
        console.log("Error in final chain", error);
        return "Error in final chain";
    }
});
const formatChatHistory = (chatHistory) => {
    console.log("Formatting Chat History");
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
    return res;
};
//# sourceMappingURL=generate.js.map