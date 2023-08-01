#!/usr/bin/env python3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


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

	def __init__(self, metadata_dir: Path | str):
		metadata_dir = Path(metadata_dir)
		self.aliases = read_tag_aliases(metadata_dir)
		self.implications = read_tag_implications(metadata_dir)
		self.blacklist = read_tag_blacklist(metadata_dir)
		self.deprecations = read_tag_deprecations(metadata_dir)

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


def read_tag_aliases(metadata_dir: Path) -> dict[str, str]:
	"""
	Returns a mapping based on tag aliases.
	This maps from aliased tags back to a canonical tag.
	Given a tag like "ff7" as key, for example, the value would be "final_fantasy_vii".
	"""
	aliases = [TagAlias.model_validate_json(line) for line in open(metadata_dir / 'tag_aliases000000000000.json', 'r')]
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


def read_tag_implications(metadata_dir: Path) -> dict[str, set[str]]:
	"""Returns a dictionary of tag implications. Given a tag like "mouse_ears" as key, for example, the value would be "animal_ears"."""
	implications = defaultdict(set)

	with open(metadata_dir / 'tag_implications000000000000.json', 'r') as f:
		for line in f:
			implication = TagImplication.model_validate_json(line)

			if implication.status != 'active':
				continue

			implications[implication.antecedent_name].add(implication.consequent_name)
	
	return implications


def read_tag_blacklist(metadata_dir: Path) -> set[str]:
	"""Returns a set of blacklisted tags."""
	with open(metadata_dir / 'tag_blacklist.txt', 'r') as f:
		return set(line.strip() for line in f.read().splitlines() if line.strip() != '')


def read_tag_deprecations(metadata_dir: Path) -> set[str]:
	"""Returns a set of deprecated tags."""
	with open(metadata_dir / 'tag_deprecations.txt', 'r') as f:
		return set(line.strip() for line in f.read().splitlines() if line.strip() != '')


def read_duplicates() -> list[set[bytes]]:
	"""Returns a list of duplicate images. Each item in the list is a `set` with a list of all hashes in that group."""
	groups = open('duplicates.txt', 'r').read().splitlines()
	groups = (group.split(' ') for group in groups)
	groups = (set(bytes.fromhex(x) for x in group) for group in groups)

	return list(groups)