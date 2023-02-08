# danbooru-metadata

Scripts and notebooks for parsing through the Danbooru2021 metadata.  For example, this includes a "top tag" list. This tag list is useful for training neural networks that, for example, predict tags for images.

Main file is `build.py`.

Note that currently this code depends on my custom Danbooru database.

Methodology:

* Go through all Danbooru metadata
* Skip posts with images that cannot be decoded into PIL.Image
   * This excludes gifs, webms, etc. as well as broken images.
   * These images cannot be trained on in other parts of the pipeline, so their tags are not useful.
* Skip tags listed in tag_blacklist.txt
   * This is a hand built list of tags that I feel aren't useful.
   * For example, tags like "bad_id" are not relavant to the image itself.
   * It's possible to generate a blacklist in an automated way from things like the metatags category (https://danbooru.donmai.us/wiki_pages/tag_group%3Ametatags), but I think manually building the blacklist, though tedious, would be more accurate.
   * I did not blacklist Tag Group: Text tags, since I believe other networks in the pipeline will be able to handle them or infer some useful information from them.
* Tag aliases are resolved to their canonical tag.
* Tag implications are applied.
   * This should help get a better tag usage count for higher order tags.
* Duplicate images are filtered using `duplicates.txt`.
   * Tags are merged for duplicate images.
   * The maximum score is used for duplicate images.
   * The maximum rating is used for duplicate images.
   * NOTE: `duplicates.txt` is generated using a perceptual hash.  While we could infer some duplicates from the Danbooru2021 metadata (child posts, etc), a perceptual hash would still be needed to catch duplicates that are not explicitly listed in the metadata.
   * The post id and file hash are picked arbitrarily for given group of duplicate images.
* After all that filtering, tag usage counts are calculated.
* Tags are sorted by usage count.
* The top 6000 tags are selected.
   * This is based on the first round number where all tags are represented at least 1000 times.
* The tag list is saved to `top_tags.txt`.
* Filtered metadata is saved to `metadata.parquet`.