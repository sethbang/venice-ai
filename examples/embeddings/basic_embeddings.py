#!/usr/bin/env python3
"""
Venice AI SDK - Basic Text Embeddings
=====================================

This example demonstrates how to generate text embeddings using the Venice AI SDK.
Learn how to convert text into vector representations for semantic analysis.
"""

import asyncio
import sys

from venice_ai import VeniceClient, cosine_similarity


async def basic_embedding_generation():
    """Generate embeddings for simple text."""
    print("🔢 Basic Embedding Generation")
    print("-" * 30)

    async with VeniceClient() as client:
        # Get available embedding model dynamically
        embedding_model = await client.models.resolve_embedding()
        print(f"📍 Using embedding model: {embedding_model}")

        # Generate embedding for a single text
        response = await client.embeddings.create(
            model=embedding_model, input="The quick brown fox jumps over the lazy dog."
        )

        # Get the embedding vector
        embedding = response.data[0].embedding

        print("✅ Generated embedding successfully")
        print(f"📏 Embedding dimensions: {len(embedding)}")
        print(f"🔢 First 5 values: {embedding[:5]}")
        print(f"📊 Usage: {response.usage.total_tokens} tokens")


async def batch_embedding_generation():
    """Generate embeddings for multiple texts at once."""
    print("\n📦 Batch Embedding Generation")
    print("-" * 30)

    # Multiple texts to embed
    texts = [
        "I love sunny weather and outdoor activities.",
        "Rainy days are perfect for reading books indoors.",
        "Machine learning is revolutionizing technology.",
        "Artificial intelligence helps solve complex problems.",
        "Pizza is my favorite food for dinner.",
    ]

    async with VeniceClient() as client:
        # Get available embedding model dynamically
        embedding_model = await client.models.resolve_embedding()
        print(f"📍 Using embedding model: {embedding_model}")

        response = await client.embeddings.create(model=embedding_model, input=texts)

        print(f"✅ Generated {len(response.data)} embeddings")
        print(f"📏 Each embedding has {len(response.data[0].embedding)} dimensions")
        print(f"📊 Total tokens used: {response.usage.total_tokens}")

        # Show the texts and their embedding previews
        for i, (text, embedding_data) in enumerate(zip(texts, response.data)):
            print(f"\n📝 Text {i + 1}: {text[:50]}...")
            print(f"🔢 Embedding preview: {embedding_data.embedding[:3]}...")


async def semantic_similarity_analysis():
    """Demonstrate semantic similarity using embeddings."""
    print("\n🎯 Semantic Similarity Analysis")
    print("-" * 30)

    # Texts with different levels of similarity
    texts = [
        "The cat sits on the mat.",
        "A feline rests on the carpet.",  # Similar to #1
        "Dogs are loyal companions.",
        "Canines make faithful friends.",  # Similar to #3
        "I enjoy eating pizza for lunch.",  # Different topic
    ]

    async with VeniceClient() as client:
        # Get available embedding model dynamically
        embedding_model = await client.models.resolve_embedding()
        print(f"📍 Using embedding model: {embedding_model}")

        response = await client.embeddings.create(model=embedding_model, input=texts)

        # The SDK ships a pure-Python cosine_similarity helper — no numpy needed.
        # ``EmbeddingObject.embedding`` is ``list[float] | str`` because the API
        # can also return base64-encoded vectors; we use the default float format.
        embeddings: list[list[float]] = [
            item.embedding for item in response.data if isinstance(item.embedding, list)
        ]

        print("🔍 Similarity Analysis (cosine similarity):")
        print("-" * 40)

        # Calculate cosine similarities between all pairs
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = cosine_similarity(embeddings[i], embeddings[j])

                print(f"\n📊 Similarity between texts {i + 1} and {j + 1}: {similarity:.3f}")
                print(f"   Text {i + 1}: {texts[i]}")
                print(f"   Text {j + 1}: {texts[j]}")

                # Interpret similarity score
                if similarity > 0.8:
                    print("   🟢 Very similar")
                elif similarity > 0.6:
                    print("   🟡 Moderately similar")
                elif similarity > 0.4:
                    print("   🟠 Somewhat similar")
                else:
                    print("   🔴 Different topics")


async def embedding_search_example():
    """Demonstrate simple semantic search using embeddings."""
    print("\n🔍 Semantic Search Example")
    print("-" * 30)

    # Document collection
    documents = [
        "Python is a popular programming language for data science.",
        "Machine learning algorithms can predict future trends.",
        "The weather today is sunny and warm.",
        "Cooking pasta requires boiling water and salt.",
        "Deep learning models use neural networks with multiple layers.",
        "Exercise is important for maintaining good health.",
        "JavaScript is commonly used for web development.",
    ]

    # Search query
    query = "artificial intelligence and programming"

    async with VeniceClient() as client:
        # Get available embedding model dynamically
        embedding_model = await client.models.resolve_embedding()
        print(f"📍 Using embedding model: {embedding_model}")

        # Embed all documents and the query
        all_texts = documents + [query]
        response = await client.embeddings.create(model=embedding_model, input=all_texts)

        # Separate document embeddings from query embedding (default float format).
        doc_embeddings: list[list[float]] = [
            item.embedding for item in response.data[:-1] if isinstance(item.embedding, list)
        ]
        raw_query = response.data[-1].embedding
        assert isinstance(raw_query, list), "expected float-format embedding"
        query_embedding: list[float] = raw_query

        # Calculate similarities (cosine_similarity returns a float in [-1, 1])
        similarities = [
            (i, cosine_similarity(query_embedding, doc)) for i, doc in enumerate(doc_embeddings)
        ]

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        print(f"🔍 Search query: '{query}'")
        print("\n📋 Most relevant documents:")

        for rank, (doc_idx, similarity) in enumerate(similarities[:3], 1):
            print(f"\n{rank}. Similarity: {similarity:.3f}")
            print(f"   📄 {documents[doc_idx]}")


async def main():
    """Run all embedding examples."""
    print("🚀 Venice AI Basic Embeddings Examples")
    print("=" * 50)

    await basic_embedding_generation()
    await batch_embedding_generation()
    await semantic_similarity_analysis()
    await embedding_search_example()

    print("\n✨ Embedding examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Single and batch embedding generation")
    print("   - Semantic similarity calculation")
    print("   - Simple semantic search")
    print("   - Vector operations with embeddings")
    print("   - Practical applications of text embeddings")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
