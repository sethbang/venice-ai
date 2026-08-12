#!/usr/bin/env python3
"""
Venice AI SDK - Batch Embedding Processing
==========================================

This example demonstrates efficient batch processing of text embeddings.
Learn how to handle large volumes of text efficiently using batching strategies.
"""

import asyncio
import sys
import time

from venice_ai import VeniceClient


async def simple_batch_embedding():
    """Demonstrate basic batch embedding of multiple texts."""
    print("📦 Simple Batch Embedding")
    print("-" * 40)

    async with VeniceClient() as client:
        # Get available embedding model
        embedding_model = await client.models.resolve_embedding()
        print(f"📍 Using embedding model: {embedding_model}")

        # Batch of texts to embed
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming technology.",
            "Python is a versatile programming language.",
            "Climate change affects global weather patterns.",
            "Quantum computing promises exponential speedups.",
        ]

        print(f"\n📝 Processing {len(texts)} texts in a single batch...")
        start_time = time.time()

        # Single batch request
        response = await client.embeddings.create(model=embedding_model, input=texts)

        elapsed = time.time() - start_time

        print(f"\n✅ Generated {len(response.data)} embeddings")
        print(f"⏱️  Time taken: {elapsed:.3f} seconds")
        print(f"📊 Total tokens: {response.usage.total_tokens}")
        print(f"⚡ Tokens/second: {response.usage.total_tokens / elapsed:.0f}")

        # Show embedding dimensions
        print("\n📏 Embedding details:")
        for i, embedding_data in enumerate(response.data[:3], 1):
            print(f"   Text {i}: {len(embedding_data.embedding)} dimensions")


async def chunked_batch_processing():
    """Demonstrate processing large datasets in chunks."""
    print("\n🔄 Chunked Batch Processing")
    print("-" * 40)

    async with VeniceClient() as client:
        # Get available embedding model
        embedding_model = await client.models.resolve_embedding()
        print(f"📍 Using embedding model: {embedding_model}")

        # Generate a larger dataset
        base_texts = [
            "Artificial intelligence and machine learning",
            "Web development with modern frameworks",
            "Database design and optimization",
            "Cloud computing infrastructure",
            "Cybersecurity best practices",
        ]

        # Duplicate to create larger dataset
        all_texts = []
        for i in range(5):
            for text in base_texts:
                all_texts.append(f"{text} - variation {i + 1}")

        print(f"\n📚 Total texts to process: {len(all_texts)}")

        # Process in chunks
        chunk_size = 10
        all_embeddings = []
        total_tokens = 0
        total_chunks = (len(all_texts) + chunk_size - 1) // chunk_size

        start_time = time.time()

        for i in range(0, len(all_texts), chunk_size):
            chunk = all_texts[i : i + chunk_size]
            chunk_num = i // chunk_size + 1

            print(f"\n📦 Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} texts)...")

            response = await client.embeddings.create(model=embedding_model, input=chunk)

            all_embeddings.extend([item.embedding for item in response.data])
            total_tokens += response.usage.total_tokens

            print(f"   ✓ Chunk {chunk_num} complete ({response.usage.total_tokens} tokens)")

        elapsed = time.time() - start_time

        print("\n✅ Batch processing complete!")
        print("📊 Statistics:")
        print(f"   Total embeddings: {len(all_embeddings)}")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Time taken: {elapsed:.3f} seconds")
        print(f"   Average time per chunk: {elapsed / total_chunks:.3f} seconds")
        print(f"   Throughput: {len(all_texts) / elapsed:.1f} texts/second")


async def concurrent_batch_processing():
    """Demonstrate concurrent processing of multiple batches."""
    print("\n⚡ Concurrent Batch Processing")
    print("-" * 40)

    async with VeniceClient() as client:
        # Get available embedding model
        embedding_model = await client.models.resolve_embedding()
        print(f"📍 Using embedding model: {embedding_model}")

        # Create multiple independent batches
        batch1 = [
            "Python programming tutorial",
            "JavaScript web development",
            "Java enterprise applications",
        ]

        batch2 = [
            "React framework basics",
            "Vue.js components",
            "Angular modules",
        ]

        batch3 = [
            "Machine learning algorithms",
            "Deep learning networks",
            "Neural network architectures",
        ]

        batches = [batch1, batch2, batch3]
        print(f"\n🔀 Processing {len(batches)} batches concurrently...")
        print(f"   Batch 1: {len(batch1)} texts (programming languages)")
        print(f"   Batch 2: {len(batch2)} texts (web frameworks)")
        print(f"   Batch 3: {len(batch3)} texts (ML topics)")

        start_time = time.time()

        # Process all batches concurrently
        tasks = [client.embeddings.create(model=embedding_model, input=batch) for batch in batches]

        responses = await asyncio.gather(*tasks)

        elapsed = time.time() - start_time

        # Collect results
        total_embeddings = sum(len(r.data) for r in responses)
        total_tokens = sum(r.usage.total_tokens for r in responses)

        print("\n✅ Concurrent processing complete!")
        print(f"⏱️  Total time: {elapsed:.3f} seconds")
        print(f"📊 Total embeddings: {total_embeddings}")
        print(f"📊 Total tokens: {total_tokens}")
        print(f"⚡ Throughput: {total_embeddings / elapsed:.1f} embeddings/second")


async def batch_with_error_handling():
    """Demonstrate batch processing with proper error handling."""
    print("\n🛡️ Batch Processing with Error Handling")
    print("-" * 40)

    async with VeniceClient() as client:
        # Get available embedding model
        embedding_model = await client.models.resolve_embedding()
        print(f"📍 Using embedding model: {embedding_model}")

        # Mix of valid and rejected inputs. The empty list is rejected
        # client-side by the SDK (embeddings.create raises InvalidRequestError
        # before any network call — see resources/embeddings.py), while the
        # huge string is sent and rejected server-side for exceeding the
        # model's context window.
        huge_input = "lorem ipsum " * 200_000  # ~2.4M chars, far over context
        batches = [
            ["Valid text 1", "Valid text 2", "Valid text 3"],
            ["Another valid text", "More valid content"],
            [],  # Empty list — SDK rejects client-side (InvalidRequestError)
            [huge_input],  # Single oversized prompt — server rejects
            ["Final batch text 1", "Final batch text 2"],
        ]

        print(f"\n📦 Processing {len(batches)} batches with error handling...")

        successful_batches = 0
        failed_batches = 0
        all_embeddings = []

        for i, batch in enumerate(batches, 1):
            try:
                print(f"\n   Batch {i}/{len(batches)}: {len(batch)} texts")

                response = await client.embeddings.create(model=embedding_model, input=batch)

                all_embeddings.extend([item.embedding for item in response.data])
                successful_batches += 1
                print(f"   ✅ Success ({len(response.data)} embeddings)")

            except Exception as e:
                failed_batches += 1
                print(f"   ❌ Error: {str(e)[:100]}")
                print("      Skipping this batch and continuing...")

        print("\n📊 Results:")
        print(f"   Successful batches: {successful_batches}")
        print(f"   Failed batches: {failed_batches}")
        print(f"   Total embeddings generated: {len(all_embeddings)}")


async def optimized_large_dataset():
    """Demonstrate optimized processing of a large dataset."""
    print("\n🚀 Optimized Large Dataset Processing")
    print("-" * 40)

    async with VeniceClient() as client:
        # Get available embedding model
        embedding_model = await client.models.resolve_embedding()
        print(f"📍 Using embedding model: {embedding_model}")

        # Simulate a large dataset
        print("\n📚 Generating large dataset...")
        large_dataset = [
            f"Document {i}: This is sample text for document number {i} in our collection."
            for i in range(50)
        ]

        print(f"   Dataset size: {len(large_dataset)} documents")

        # Optimized processing strategy
        chunk_size = 20
        max_concurrent = 3

        print("\n⚙️  Processing strategy:")
        print(f"   Chunk size: {chunk_size}")
        print(f"   Max concurrent batches: {max_concurrent}")

        # Split into chunks
        chunks = [
            large_dataset[i : i + chunk_size] for i in range(0, len(large_dataset), chunk_size)
        ]

        all_embeddings = []
        total_tokens = 0
        start_time = time.time()

        # Process chunks in groups of max_concurrent
        for group_start in range(0, len(chunks), max_concurrent):
            group_chunks = chunks[group_start : group_start + max_concurrent]
            group_num = group_start // max_concurrent + 1
            total_groups = (len(chunks) + max_concurrent - 1) // max_concurrent

            print(
                f"\n🔄 Processing group {group_num}/{total_groups} ({len(group_chunks)} batches)..."
            )

            # Process this group concurrently
            tasks = [
                client.embeddings.create(model=embedding_model, input=chunk)
                for chunk in group_chunks
            ]

            responses = await asyncio.gather(*tasks)

            # Collect results
            for response in responses:
                all_embeddings.extend([item.embedding for item in response.data])
                total_tokens += response.usage.total_tokens

            print(f"   ✓ Group {group_num} complete")

        elapsed = time.time() - start_time

        print("\n✅ Large dataset processing complete!")
        print("📊 Final statistics:")
        print(f"   Total documents: {len(large_dataset)}")
        print(f"   Total embeddings: {len(all_embeddings)}")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Time taken: {elapsed:.3f} seconds")
        print(f"   Throughput: {len(large_dataset) / elapsed:.1f} docs/second")
        print(f"   Average tokens per doc: {total_tokens / len(large_dataset):.1f}")


async def batch_deduplication():
    """Demonstrate batch processing with duplicate detection."""
    print("\n🔍 Batch Processing with Deduplication")
    print("-" * 40)

    async with VeniceClient() as client:
        # Get available embedding model
        embedding_model = await client.models.resolve_embedding()
        print(f"📍 Using embedding model: {embedding_model}")

        # Dataset with some duplicates
        texts = [
            "Machine learning fundamentals",
            "Python programming basics",
            "Machine learning fundamentals",  # Duplicate
            "Web development with React",
            "Python programming basics",  # Duplicate
            "Database optimization techniques",
            "Cloud computing platforms",
        ]

        print(f"\n📚 Original dataset: {len(texts)} texts")

        # Deduplicate before processing
        unique_texts = list(dict.fromkeys(texts))  # Preserves order
        duplicates_removed = len(texts) - len(unique_texts)

        print(f"🔍 After deduplication: {len(unique_texts)} unique texts")
        print(f"   Removed {duplicates_removed} duplicates")

        # Process only unique texts
        print(f"\n📦 Processing {len(unique_texts)} unique texts...")

        response = await client.embeddings.create(model=embedding_model, input=unique_texts)

        print("\n✅ Processing complete!")
        print(f"📊 Embeddings generated: {len(response.data)}")
        print(f"💰 Tokens saved by deduplication: ~{duplicates_removed * 5}")  # Rough estimate
        print(f"📊 Actual tokens used: {response.usage.total_tokens}")


async def main():
    """Run all batch processing examples."""
    print("🚀 Venice AI Batch Embedding Processing Examples")
    print("=" * 50)

    await simple_batch_embedding()
    await chunked_batch_processing()
    await concurrent_batch_processing()
    await batch_with_error_handling()
    await optimized_large_dataset()
    await batch_deduplication()

    print("\n✨ Batch processing examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Simple batch embedding")
    print("   - Chunked processing for large datasets")
    print("   - Concurrent batch processing")
    print("   - Error handling in batches")
    print("   - Optimized large-scale processing")
    print("   - Deduplication for efficiency")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
