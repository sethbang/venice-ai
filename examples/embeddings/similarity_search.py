#!/usr/bin/env python3
"""
Venice AI SDK - Semantic Similarity Search
==========================================

This example demonstrates how to perform semantic similarity searches using embeddings.
Learn how to build a simple search engine that understands meaning, not just keywords.
"""

import asyncio
import math
import sys
from collections.abc import Sequence

from venice_ai import VeniceClient


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors (pure stdlib).

    Equivalent to numpy's ``dot(a, b) / (norm(a) * norm(b))`` but uses only
    :mod:`math`, so the example runs on a clean ``pip install venice-ai``
    without numpy (which is a dev-only dependency).
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


# Sample document collection for search examples
TECH_ARTICLES = [
    "Python is a high-level programming language known for its simplicity and readability.",
    "JavaScript powers interactive web applications and runs in the browser.",
    "Machine learning algorithms can identify patterns in large datasets.",
    "Deep neural networks are inspired by biological neural networks in the brain.",
    "Cloud computing provides on-demand access to computing resources over the internet.",
    "Blockchain technology enables secure, decentralized transaction records.",
    "Quantum computers use quantum bits (qubits) for exponentially faster calculations.",
    "Cybersecurity protects systems and networks from digital attacks.",
    "Data science combines statistics, programming, and domain knowledge.",
    "Artificial intelligence aims to create machines that can perform human-like tasks.",
]

COOKING_RECIPES = [
    "Spaghetti carbonara is made with eggs, cheese, pancetta, and black pepper.",
    "Chocolate chip cookies require butter, sugar, eggs, flour, and chocolate chips.",
    "A classic Caesar salad includes romaine lettuce, croutons, and parmesan cheese.",
    "Thai green curry combines coconut milk, green curry paste, and vegetables.",
    "Homemade pizza dough needs flour, yeast, water, salt, and olive oil.",
]


async def basic_similarity_search():
    """Demonstrate basic semantic search across documents."""
    print("🔍 Basic Similarity Search")
    print("-" * 40)

    async with VeniceClient() as client:
        # Get available embedding model
        embedding_model = await client.models.resolve_embedding()
        print(f"📍 Using embedding model: {embedding_model}")

        # Embed all documents
        print(f"\n📚 Embedding {len(TECH_ARTICLES)} technology articles...")
        response = await client.embeddings.create(model=embedding_model, input=TECH_ARTICLES)

        doc_embeddings = [item.embedding for item in response.data]

        # Search query
        query = "artificial intelligence and neural networks"
        print(f"\n🔎 Search query: '{query}'")

        # Embed the query
        query_response = await client.embeddings.create(model=embedding_model, input=query)
        query_embedding = query_response.data[0].embedding

        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(doc_embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Display top results
        print("\n📊 Top 3 most relevant articles:")
        for rank, (doc_idx, similarity) in enumerate(similarities[:3], 1):
            print(f"\n{rank}. Similarity: {similarity:.4f}")
            print(f"   📄 {TECH_ARTICLES[doc_idx]}")


async def multi_query_search():
    """Demonstrate searching with multiple queries."""
    print("\n🔎 Multi-Query Search")
    print("-" * 40)

    async with VeniceClient() as client:
        # Get available embedding model
        embedding_model = await client.models.resolve_embedding()
        print(f"📍 Using embedding model: {embedding_model}")

        # Embed documents
        doc_response = await client.embeddings.create(model=embedding_model, input=TECH_ARTICLES)
        doc_embeddings = [item.embedding for item in doc_response.data]

        # Multiple search queries
        queries = [
            "programming languages",
            "machine learning and AI",
            "web development",
        ]

        # Embed all queries at once
        query_response = await client.embeddings.create(model=embedding_model, input=queries)
        query_embeddings = [item.embedding for item in query_response.data]

        # Search for each query
        for query, query_embedding in zip(queries, query_embeddings):
            print(f"\n🔍 Query: '{query}'")

            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(doc_embeddings):
                similarity = cosine_similarity(query_embedding, doc_embedding)
                similarities.append((i, similarity))

            # Get top result
            similarities.sort(key=lambda x: x[1], reverse=True)
            doc_idx, similarity = similarities[0]

            print(f"   Best match ({similarity:.4f}): {TECH_ARTICLES[doc_idx][:60]}...")


async def threshold_filtering():
    """Demonstrate filtering results by similarity threshold."""
    print("\n🎯 Threshold-Based Filtering")
    print("-" * 40)

    async with VeniceClient() as client:
        # Get available embedding model
        embedding_model = await client.models.resolve_embedding()
        print(f"📍 Using embedding model: {embedding_model}")

        # Embed documents
        doc_response = await client.embeddings.create(model=embedding_model, input=TECH_ARTICLES)
        doc_embeddings = [item.embedding for item in doc_response.data]

        # Query
        query = "cooking recipes"
        print(f"\n🔍 Query: '{query}'")
        print("   (Note: This query is about cooking, but our docs are about tech)")

        query_response = await client.embeddings.create(model=embedding_model, input=query)
        query_embedding = query_response.data[0].embedding

        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(doc_embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        # Apply different thresholds. These bracket bge-m3's actual cosine range
        # for this off-topic query (~0.35-0.46), so each cutoff filters a
        # progressively larger slice of the docs instead of all-or-nothing.
        thresholds = [0.44, 0.40, 0.36]

        for threshold in thresholds:
            print(f"\n📏 Threshold: {threshold}")
            results = [(idx, sim) for idx, sim in similarities if sim >= threshold]

            if results:
                print(f"   Found {len(results)} result(s):")
                for doc_idx, similarity in results[:3]:  # Show max 3
                    print(f"   - ({similarity:.4f}) {TECH_ARTICLES[doc_idx][:50]}...")
            else:
                print(f"   ⚠️ No results above threshold {threshold}")


async def cross_domain_search():
    """Demonstrate searching across different document collections."""
    print("\n🌐 Cross-Domain Search")
    print("-" * 40)

    async with VeniceClient() as client:
        # Get available embedding model
        embedding_model = await client.models.resolve_embedding()
        print(f"📍 Using embedding model: {embedding_model}")

        # Combine both collections
        all_docs = TECH_ARTICLES + COOKING_RECIPES
        doc_types = ["tech"] * len(TECH_ARTICLES) + ["cooking"] * len(COOKING_RECIPES)

        print("\n📚 Document collection:")
        print(f"   Tech articles: {len(TECH_ARTICLES)}")
        print(f"   Cooking recipes: {len(COOKING_RECIPES)}")
        print(f"   Total: {len(all_docs)}")

        # Embed all documents
        doc_response = await client.embeddings.create(model=embedding_model, input=all_docs)
        doc_embeddings = [item.embedding for item in doc_response.data]

        # Test with different queries
        queries = [
            "how to make pasta",
            "artificial intelligence",
        ]

        query_response = await client.embeddings.create(model=embedding_model, input=queries)

        for query, query_data in zip(queries, query_response.data):
            query_embedding = query_data.embedding

            print(f"\n🔍 Query: '{query}'")

            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(doc_embeddings):
                similarity = cosine_similarity(query_embedding, doc_embedding)
                similarities.append((i, similarity, doc_types[i]))

            # Sort and show top results
            similarities.sort(key=lambda x: x[1], reverse=True)

            print("   Top 3 results:")
            for rank, (doc_idx, similarity, doc_type) in enumerate(similarities[:3], 1):
                emoji = "👨‍💻" if doc_type == "tech" else "👨‍🍳"
                print(f"   {rank}. {emoji} [{doc_type}] ({similarity:.4f})")
                print(f"      {all_docs[doc_idx][:60]}...")


async def ranked_search_with_metadata():
    """Demonstrate search with result ranking and metadata."""
    print("\n📊 Ranked Search with Metadata")
    print("-" * 40)

    # Document metadata
    doc_metadata = [
        {"id": 1, "category": "language", "difficulty": "beginner"},
        {"id": 2, "category": "language", "difficulty": "beginner"},
        {"id": 3, "category": "ml", "difficulty": "intermediate"},
        {"id": 4, "category": "ml", "difficulty": "advanced"},
        {"id": 5, "category": "infrastructure", "difficulty": "intermediate"},
        {"id": 6, "category": "infrastructure", "difficulty": "advanced"},
        {"id": 7, "category": "hardware", "difficulty": "advanced"},
        {"id": 8, "category": "security", "difficulty": "intermediate"},
        {"id": 9, "category": "data", "difficulty": "intermediate"},
        {"id": 10, "category": "ml", "difficulty": "intermediate"},
    ]

    async with VeniceClient() as client:
        # Get available embedding model
        embedding_model = await client.models.resolve_embedding()
        print(f"📍 Using embedding model: {embedding_model}")

        # Embed documents
        doc_response = await client.embeddings.create(model=embedding_model, input=TECH_ARTICLES)
        doc_embeddings = [item.embedding for item in doc_response.data]

        # Query
        query = "learn about AI"
        print(f"\n🔍 Query: '{query}'")

        query_response = await client.embeddings.create(model=embedding_model, input=query)
        query_embedding = query_response.data[0].embedding

        # Calculate similarities and combine with metadata
        results = []
        for i, doc_embedding in enumerate(doc_embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            results.append(
                {
                    "index": i,
                    "similarity": similarity,
                    "text": TECH_ARTICLES[i],
                    "metadata": doc_metadata[i],
                }
            )

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # Display results with metadata
        print("\n📋 Top 5 results with metadata:")
        for rank, result in enumerate(results[:5], 1):
            meta = result["metadata"]
            print(f"\n{rank}. Similarity: {result['similarity']:.4f}")
            print(f"   📄 {result['text'][:60]}...")
            print(f"   🏷️  Category: {meta['category']} | Difficulty: {meta['difficulty']}")


async def find_similar_documents():
    """Find documents similar to a given document (document-to-document similarity)."""
    print("\n📄 Document-to-Document Similarity")
    print("-" * 40)

    async with VeniceClient() as client:
        # Get available embedding model
        embedding_model = await client.models.resolve_embedding()
        print(f"📍 Using embedding model: {embedding_model}")

        # Embed all documents
        doc_response = await client.embeddings.create(model=embedding_model, input=TECH_ARTICLES)
        doc_embeddings = [item.embedding for item in doc_response.data]

        # Select a reference document
        reference_idx = 3  # Deep neural networks article
        reference_doc = TECH_ARTICLES[reference_idx]
        reference_embedding = doc_embeddings[reference_idx]

        print("\n📌 Reference document:")
        print(f"   {reference_doc}")

        # Find similar documents
        similarities = []
        for i, doc_embedding in enumerate(doc_embeddings):
            if i == reference_idx:
                continue  # Skip the reference document itself

            similarity = cosine_similarity(reference_embedding, doc_embedding)
            similarities.append((i, similarity))

        # Sort and display
        similarities.sort(key=lambda x: x[1], reverse=True)

        print("\n🔍 Most similar documents:")
        for rank, (doc_idx, similarity) in enumerate(similarities[:3], 1):
            print(f"\n{rank}. Similarity: {similarity:.4f}")
            print(f"   📄 {TECH_ARTICLES[doc_idx]}")


async def main():
    """Run all similarity search examples."""
    print("🚀 Venice AI Semantic Similarity Search Examples")
    print("=" * 50)

    await basic_similarity_search()
    await multi_query_search()
    await threshold_filtering()
    await cross_domain_search()
    await ranked_search_with_metadata()
    await find_similar_documents()

    print("\n✨ Similarity search examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Basic semantic search")
    print("   - Multi-query batch processing")
    print("   - Threshold-based filtering")
    print("   - Cross-domain document search")
    print("   - Ranked results with metadata")
    print("   - Document-to-document similarity")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
