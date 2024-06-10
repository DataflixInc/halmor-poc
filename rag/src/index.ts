import express, { Request, Response } from "express";
import dotenv from "dotenv";
import { generate } from "./generate";

dotenv.config();

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 8080;

/*
    - POST /generate
        - Body
            - question: Careers URL to scrape Job Description

        - Response

    What's happening in this API?
*/
app.post("/generate", async (req: Request, res: Response) => {
    try {
        let body = req.body;

        if (body.chatHistory) {
            const answer = await generate(body.chatHistory);
            res.send({
                response: {
                    answer,
                },
            });
        } else {
            res.send({
                response: "No response generated",
            });
        }
    } catch (error) {
        console.log({ error });
        res.send({
            response: "Error generating quiz",
        });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
