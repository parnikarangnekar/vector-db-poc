package com.example;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.pgvector.PgVectorEmbeddingStore;

import java.io.IOException;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            System.out.println("Usage:");
            System.out.println("  ./gradlew run --args=\"ingest\"       <- Embeds content from documentation.md");
            System.out.println("  ./gradlew run --args=\"search 'query'\" <- Searches for content");
            System.out.println("  ./gradlew run --args=\"chat 'question'\" <- Asks the AI Agent");
            System.out.println("  ./gradlew run --args=\"reset\"        <- Clears the database");
            return;
        }

        String command = args[0];

        // 1. Initialize the Embedding Model (running locally)
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // 2. Initialize the Embedding Store
        EmbeddingStore<TextSegment> embeddingStore = PgVectorEmbeddingStore.builder()
                .host("localhost")
                .port(5432)
                .database("")
                .user("")
                .password("")
                .table("")
                .dimension(384)
                .createTable(true)
                .build();

        if ("ingest".equalsIgnoreCase(command)) {
            System.out.println("--- Ingest Mode ---");
            if (args.length > 1) {
                for (int i = 1; i < args.length; i++) {
                    String startPath = args[i];
            System.out.println("Ingesting from: " + startPath);

            java.nio.file.Path path = java.nio.file.Paths.get(startPath);
            if (!java.nio.file.Files.exists(path)) {
                System.out.println("Path does not exist: " + startPath);
                return;
            }

                    DocumentSplitter splitter = DocumentSplitters.recursive(512, 100);

            try (java.util.stream.Stream<java.nio.file.Path> paths = java.nio.file.Files.walk(path)) {
                paths.filter(java.nio.file.Files::isRegularFile)
                        .filter(p -> p.toString().endsWith(".md") || p.toString().endsWith(".txt"))
                        .forEach(file -> {
                            try {
                                System.out.println("Processing: " + file);
                                String text = new String(java.nio.file.Files.readAllBytes(file));
                                if (text.trim().isEmpty())
                                    return;

                                Document document = Document.from(text);
                                List<TextSegment> segments = splitter.split(document);
                                if (segments.isEmpty())
                                    return;

                                System.out.println("  - Embedding " + segments.size() + " segments...");
                                List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
                                embeddingStore.addAll(embeddings, segments);
                            } catch (Exception e) {
                                System.out.println("Failed to process file " + file + ": " + e.getMessage());
                            }
                        });
            } catch (IOException e) {
                System.out.println("Error walking files: " + e.getMessage());
            }
                }
            } else {
                System.out.println("Missing Arguments");
            }
        } else if ("reset".equalsIgnoreCase(command)) {
            System.out.println("--- Reset Mode ---");
            try (java.sql.Connection conn = java.sql.DriverManager
                    .getConnection("jdbc:postgresql://localhost:5432/<database_name>", "", "");
                    java.sql.Statement stmt = conn.createStatement()) {
                stmt.execute("TRUNCATE TABLE <table_name>");
                System.out.println("Table 'documents' cleared.");
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else if ("chat".equalsIgnoreCase(command)) {
            if (args.length < 2) {
                System.out.println("Please provide a question. Example: ./gradlew run --args=\"chat 'How do I...?'\"");
                return;
            }
            String question = args[1];

            // 1. Retrieve relevant segments
            Embedding questionEmbedding = embeddingModel.embed(question).content();
            List<EmbeddingMatch<TextSegment>> relevant = embeddingStore.findRelevant(questionEmbedding, 5, 0.6);

            if (relevant.isEmpty()) {
                System.out.println("I don't have enough information to answer that based on the documentation.");
                return;
            }

            String context = relevant.stream()
                    .map(match -> match.embedded().text())
                    .collect(java.util.stream.Collectors.joining("\n\n"));

            // 2. Construct Prompt
            String prompt = "You are a helpful assistant for the HotWax Commerce Order Routing system.\n" +
                    "Answer the user's question based strictly on the context provided below.\n" +
                    "If the answer is not in the context, say 'I don't know'.\n\n" +
                    "Context:\n" + context + "\n\n" +
                    "Question: " + question;

            // 3. Call Gemini
            String apiKey = System.getenv("GOOGLE_AI_GEMINI_API_KEY");
            if (apiKey == null || apiKey.isEmpty()) {
                System.out.println("Please set GOOGLE_AI_GEMINI_API_KEY environment variable.");
                return;
            }

            dev.langchain4j.model.chat.ChatLanguageModel chatModel = dev.langchain4j.model.googleai.GoogleAiGeminiChatModel
                    .builder()
                    .apiKey(apiKey)
                    .modelName("models/gemini-pro")
                    .build();

            try {
                String response = chatModel.generate(prompt);
                System.out.println("\nAnswer:");
                System.out.println(response);
            } catch (Exception e) {
                System.out.println("\nError calling Gemini API: " + e.getMessage());
                System.out.println("Please check if your API Key is valid and has access to 'gemini-1.5-flash'.");
                System.out.println("You can generate a new key at: https://aistudio.google.com/");
            }

        } else {
            System.out.println("Unknown command: " + command);
        }
    }
}
