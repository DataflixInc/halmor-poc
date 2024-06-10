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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const dotenv_1 = __importDefault(require("dotenv"));
const generate_1 = require("./generate");
dotenv_1.default.config();
const app = (0, express_1.default)();
app.use(express_1.default.json());
const PORT = process.env.PORT || 8080;
/*
    - POST /generate
        - Body
            - question: Careers URL to scrape Job Description

        - Response

    What's happening in this API?
*/
app.post("/generate", (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        let body = req.body;
        if (body.chatHistory) {
            const answer = yield (0, generate_1.generate)(body.chatHistory);
            res.send({
                response: {
                    answer,
                },
            });
        }
        else {
            res.send({
                response: "No response generated",
            });
        }
    }
    catch (error) {
        console.log({ error });
        res.send({
            response: "Error generating quiz",
        });
    }
}));
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
//# sourceMappingURL=index.js.map