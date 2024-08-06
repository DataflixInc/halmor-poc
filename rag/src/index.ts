import express, { Request, Response } from "express";
import dotenv from "dotenv";
import { generate } from "./generate";

dotenv.config();

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 8080;

app.post("/generate", async (req: Request, res: Response) => {
    try {
        let body = req.body;

        if (body.question) {
            // Generate response based on Chat History of the user from the Watsonx bot
            const answer = await generate(body.question);
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
