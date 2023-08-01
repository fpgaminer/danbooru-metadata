#!/usr/bin/env python3
from itertools import islice
from pathlib import Path
from typing import Any, Tuple

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from danbooru_metadata import TagMappings, PostInfo, RATING_MAP, read_duplicates


# Schema for the metadata file
schema = pa.schema([
	pa.field("post_id", pa.int64()),
	pa.field("tags", pa.list_(pa.int16())),
	pa.field("hash", pa.binary(32)),  # sha256 hash of the source file
	pa.field("rating", pa.int8()),
	pa.field("score", pa.int16()),
])


def main():
	db = get_db_connection()

	print("Reading tag aliases, implications, etc...")
	tag_mappings = TagMappings()
	duplicate_groups = read_duplicates()

	# Build a metadata dictionary from the database
	metadata = build_metadata(db, tag_mappings, duplicate_groups)

	# Count tags
	tag_counts = count_tags(metadata)

	# Sort tags by count
	tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

	print(f"Found {len(tags)} tags")
	print(f"Found {len([tag for tag, count in tags if count >= 10000])} tags with at least 10,000 usage")
	print(f"Found {len([tag for tag, count in tags if count >= 1000])} tags with at least 1,000 usage")

	# Top 6000 tags
	top_tags = [tag for tag,_ in tags[:6000]]

	# Write top tags to file
	with open('top_tags.txt', 'w') as f:
		for tag in top_tags:
			f.write(tag + '\n')
	
	# Useful assertions
	assert 'safe' not in top_tags
	assert 'questionable' not in top_tags
	assert 'nsfw' not in top_tags
	assert 'worst_quality' not in top_tags
	assert 'low_quality' not in top_tags
	assert 'medium_quality' not in top_tags
	assert 'high_quality' not in top_tags
	assert 'best_quality' not in top_tags
	assert 'masterpiece' not in top_tags

	# Write metadata to file
	write_metadata_parquet(metadata, top_tags)


def build_metadata(
	db: psycopg.Connection[Tuple[Any, ...]],
	tag_mappings: TagMappings,
	duplicate_groups: list[set[bytes]],
) -> dict[int, PostInfo]:
	"""
	Build a metadata dictionary from the database.
	Each entry in the dictionary is a PostInfo object.
	The key is the post_id of the post.
	Duplicate images will be merged into a single entry.
	RATING_MAP is used to convert the rating string to an integer.
	tag_string is split into a set of tags.
	Tag aliases are applied to canonicalize tags.
	Tag implications are applied to expand tags and make sure general tags are counted correctly.
	For example, "mouse_ears" is an implication of "animal_ears", so if a post has "mouse_ears" it should be counted as having "animal_ears" as well.
	"""
	hash_to_duplicates_group_id = {hash: i for i, group in enumerate(duplicate_groups) for hash in group}

	print("Counting posts...")
	with db.cursor() as cur:
		# Only count posts with embeddings (this excludes gif posts, for example)
		cur.execute("SELECT COUNT(*) FROM metadata INNER JOIN embeddings ON metadata.file_hash = embeddings.hash")
		result = cur.fetchone()
		assert result is not None
		total_posts, = result
	
	group_id_to_post_id: dict[int, int] = {}
	metadata: dict[int, PostInfo] = {}
	
	print("Building metadata...")
	with db.cursor("metadata_query") as cur:
		cur.execute("SELECT metadata.post_id, metadata.tag_string, metadata.file_hash, metadata.score, metadata.rating FROM metadata INNER JOIN embeddings ON metadata.file_hash = embeddings.hash")

		for row in tqdm(cur, total=total_posts):
			post = PostInfo(
				post_id=row[0],
				tags=set(t.strip() for t in row[1].split(' ')),
				hash=bytes(row[2]),
				score=row[3],
				rating=RATING_MAP[row[4]],
			)

			# Apply tag aliases
			post.tags = set(tag_mappings.get_canonical(tag) for tag in post.tags)

			# Apply tag implications
			for tag in list(post.tags):
				post.tags.update(tag_mappings.get_implications(tag))

			# Remove blacklisted tags
			post.tags.difference_update(tag_mappings.blacklist)

			# Remove deprecated tags
			post.tags.difference_update(tag_mappings.deprecations)

			# Combine the data for duplicates
			# Groups of duplicate images will be merged into a single entry in metadata
			# The post_id of the first encountered image in a group will be used as the key
			# The tags of all images in the group will be merged
			# The highest score and rating will be used
			if post.hash in hash_to_duplicates_group_id:
				group_id = hash_to_duplicates_group_id[post.hash]

				if group_id not in group_id_to_post_id:
					group_id_to_post_id[group_id] = post.post_id
					metadata[post.post_id] = post
				
				group_post = metadata[group_id_to_post_id[group_id]]
				group_post.tags.update(post.tags)
				group_post.score = max(group_post.score, post.score)
				group_post.rating = max(group_post.rating, post.rating)
			else:
				metadata[post.post_id] = post
	
	print(f"{total_posts - len(metadata)} duplicate posts removed")

	return metadata


def count_tags(metadata) -> dict[str, int]:
	"""
	Count the number of times each tag appears in the metadata.
	"""
	tag_counts = {}
	max_tags = 0
	min_tags = 999999999
	sum_tags = 0

	for post in metadata.values():
		for tag in post.tags:
			tag_counts[tag] = tag_counts.get(tag, 0) + 1
		
		min_tags = min(min_tags, len(post.tags))
		max_tags = max(max_tags, len(post.tags))
		sum_tags += len(post.tags)

	mean_tags = sum_tags / len(metadata)

	print(f"Min tags: {min_tags}")
	print(f"Max tags: {max_tags}")
	print(f"Mean tags: {mean_tags}")

	return tag_counts


def write_metadata_parquet(metadata: dict[int, PostInfo], top_tags: list[str]) -> None:
	"""
	Write the metadata to a Parquet file.
	The tags field of each PostInfo object is converted to a list of integers, based on the top_tags list.
	"""
	top_tags_map = {tag: i for i, tag in enumerate(top_tags)}

	def transform(post: PostInfo) -> Tuple[int, list[int], bytes, int, int]:
		return (
			post.post_id,
			[top_tags_map[tag] for tag in post.tags if tag in top_tags_map],
			post.hash,
			post.rating,
			post.score,
		)
	
	posts = (transform(post) for post in metadata.values())
	posts = batcher(posts, 1000)

	with pq.ParquetWriter("metadata.parquet", schema) as writer:
		for batch in posts:
			post_ids, tags, hashes, ratings, scores = zip(*batch)

			batch = pa.RecordBatch.from_arrays([
				pa.array(post_ids, type=pa.int64()),
				pa.array(tags, type=pa.list_(pa.int16())),
				pa.array(hashes, type=pa.binary(32)),
				pa.array(ratings, type=pa.int8()),
				pa.array(scores, type=pa.int16()),
			], schema=schema)
			writer.write(batch)


def get_db_connection() -> psycopg.Connection[Tuple[Any, ...]]:
	return psycopg.connect("dbname=postgres user=postgres", host=str((Path.cwd() / ".." / "pg-socket").absolute()))


def batcher(iterable, n):
	iterator = iter(iterable)
	while batch := list(islice(iterator, n)):
		yield batch


if __name__ == '__main__':
	main()