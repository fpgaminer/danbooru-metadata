#!/usr/bin/env python3
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel
from tqdm.auto import tqdm


RATING_MAP = {
	's': 0,
	'q': 1,
	'e': 2,
}


@dataclass
class PostInfo:
	post_id: int
	tags: set[str]
	hash: bytes
	score: int
	rating: int


class TagAlias(BaseModel):
	antecedent_name: str
	consequent_name: str
	status: str


class TagImplication(BaseModel):
	antecedent_name: str
	consequent_name: str
	status: str


class TagMappings:
	aliases: dict[str, str]
	implications: dict[str, set[str]]
	blacklist: set[str]
	deprecations: set[str]

	def __init__(self):
		self.aliases = read_tag_aliases()
		self.implications = read_tag_implications()
		self.blacklist = read_tag_blacklist()
		self.deprecations = read_tag_deprecations()

		# Canonicalize tag implications by applying tag aliases
		self.implications = {
			self.get_canonical(tag): set(self.get_canonical(implied_tag) for implied_tag in implied_tags)
			for tag, implied_tags in self.implications.items()
		}

		# Expand tag implications
		# This condenses chains of implications into a single mapping
		# For example, if "a" implies "b" and "b" implies "c", then "a" implies "b" and "c"
		while True:
			implication_updates = {}

			for tag, implied_tags in self.implications.items():
				new_implications = (self.get_implications(implied_tag) for implied_tag in implied_tags)
				new_implications = set.union(*new_implications)
				new_implications = new_implications.difference(implied_tags)

				if len(new_implications) > 0:
					implication_updates[tag] = new_implications
			
			if len(implication_updates) == 0:
				break

			for tag, new_implications in implication_updates.items():
				self.implications[tag].update(new_implications)
	
	def get_canonical(self, tag: str) -> str:
		"""Returns the canonical name for a tag, based on aliases."""
		return self.aliases.get(tag, tag)
	
	def get_implications(self, tag: str) -> set[str]:
		"""Returns a set of all tags implied by the given tag."""
		tag = self.get_canonical(tag)
		return self.implications.get(tag, set())


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
) -> Dict[int, PostInfo]:
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


def count_tags(metadata) -> Dict[str, int]:
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


def write_metadata_parquet(metadata: Dict[int, PostInfo], top_tags: List[str]) -> None:
	"""
	Write the metadata to a Parquet file.
	The tags field of each PostInfo object is converted to a list of integers, based on the top_tags list.
	"""
	top_tags_map = {tag: i for i, tag in enumerate(top_tags)}

	def transform(post: PostInfo) -> Tuple[int, List[int], bytes, int, int]:
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


def read_tag_aliases() -> Dict[str, str]:
	"""
	Returns a mapping based on tag aliases.
	This maps from aliased tags back to a canonical tag.
	Given a tag like "ff7" as key, for example, the value would be "final_fantasy_vii".
	"""
	aliases = [TagAlias.model_validate_json(line) for line in open('../metadata/tag_aliases000000000000.json', 'r')]
	alias_map = {}

	for alias in aliases:
		if alias.status != 'active':
			continue

		assert alias.antecedent_name != alias.consequent_name, "Self-aliases found in tag aliases"

		# Duplicate antecedent->consequent mappings are allowed, but only if they are the same
		# This is because the dataset contains a few duplicates (unknown why)
		assert alias.antecedent_name not in alias_map or alias_map[alias.antecedent_name] == alias.consequent_name, "Duplicate antecedents found in tag aliases"

		alias_map[alias.antecedent_name] = alias.consequent_name

	# Check for chains by ensuring that consequents are not also antecedents
	assert all(consequent not in alias_map for consequent in alias_map.values()), "Chains found in tag aliases"
	
	return alias_map


def read_tag_implications() -> Dict[str, Set[str]]:
	"""Returns a dictionary of tag implications. Given a tag like "mouse_ears" as key, for example, the value would be "animal_ears"."""
	implications = defaultdict(set)

	with open('../metadata/tag_implications000000000000.json', 'r') as f:
		for line in f:
			implication = TagImplication.model_validate_json(line)

			if implication.status != 'active':
				continue

			implications[implication.antecedent_name].add(implication.consequent_name)
	
	return implications


def read_tag_blacklist() -> Set[str]:
	"""Returns a set of blacklisted tags."""
	with open('tag_blacklist.txt', 'r') as f:
		return set(line.strip() for line in f.read().splitlines() if line.strip() != '')


def read_tag_deprecations() -> Set[str]:
	"""Returns a set of deprecated tags."""
	with open('tag_deprecations.txt', 'r') as f:
		return set(line.strip() for line in f.read().splitlines() if line.strip() != '')


def read_duplicates() -> List[Set[bytes]]:
	"""Returns a list of duplicate images. Each item in the list is a `set` with a list of all hashes in that group."""
	groups = open('duplicates.txt', 'r').read().splitlines()
	groups = (group.split(' ') for group in groups)
	groups = (set(bytes.fromhex(x) for x in group) for group in groups)

	return list(groups)


def batcher(iterable, n):
	iterator = iter(iterable)
	while batch := list(islice(iterator, n)):
		yield batch


if __name__ == '__main__':
	main()