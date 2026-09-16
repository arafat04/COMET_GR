### confidence_score_solved_final.ipynb:

In[2]: works with: postedition_aligned.community.tsv - this file contains the aligned source to its translation and postediton for community post editors.

id_hal	Translation_id	Postedit_id	line_id	source	translation	postedition
1988871	7033	402	0	Transforming Dependency Structures to LTAG Derivation Trees	Transformation de structures de dépendances en arbres de dérivation LTAG	Transformation de structures de dépendances en arbres de dérivation LTAG

In[8]: groupby("Postedit_id").agg(...)

This **groups by only Postedit_id**, meaning it collapses all the lines belonging to the same document into a single row. For each document, it collects:

every source sentence, in original row order, into one list
every translation sentence into one list
every postedition sentence into one list

So instead of one row per sentence/line, you now get one row per document, where each cell holds the ordered list of all sentences for that document. My grouped_df.shape of (95, 4) confirms this — I went from many line-level rows down to 95 documents (one row per unique Postedit_id), each with 4 columns: Postedit_id, and the three list-valued columns.

- [ ] **which versions of df I am working with**

In[17]: 

1. Point directly to a hardcoded local file path — the exact checkpoint file location inside the Hugging Face cache folder (snapshots/<hash>/checkpoints/model.ckpt), presumably from a previous download that already happened on disk.
2. Call load_from_checkpoint(path) — the plain COMET function (not the custom CustomXCOMET subclass) — to load the model straight from that known local path, skipping download_model() entirely since the file is assumed to already exist locally.

In[20]:

1. download_model() doesn't blindly re-download every time. Under the hood it uses Hugging Face Hub's caching mechanism (huggingface_hub.snapshot_download or similar), which:

2. Checks the local cache directory (~/.cache/huggingface/hub/...) for the repo.
If the files are already there and match the expected revision/hash, it skips downloading the actual file contents.
3. It returns the local path to the cached checkpoint, exactly like it would on a fresh download.

So functionally: no bytes get re-downloaded,however it still does a cache lookup and returns the path. That's actually consistent with the notebook's output:

Fetching 5 files: 100%|██████████| 5/5 [00:00<00:00, 7145.32it/s]

**confidence_score_solved.ipynb is the initial version of confidence_score_solved_final.ipynb**

confidence_score_solved_final.ipynb: 

1. is working with postedition_aligned.community.tsv
2. loading the model from already downloaded checkpoints and instantiating the CustomXCOMET with the checkpoint.
3. The model can generate spans with or without references. Here we have the reference: its doing this on cell In [7] with the heading: **create data list for calculating the spans without the ref** and dumping the result in this file: **output_all_spans_community.json** in In [22].

       data = [
        {
          "src": "Boris Johnson teeters on edge of favour with Tory MPs",
          "mt": "Boris Johnsons Beliebtheit bei Tory-Abgeordneten völlig in der Gunst",
          "ref": "Boris Johnsons Beliebtheit bei Tory-MPs steht auf der Kippe"
        }
        ]
so it expects data in a list of dictionaries where each keys needed to be explicitly specified.

So we need to create the list of dictionaries from the dataframe. so in **cell In[7]**, it is creating a list of dicts from the **original** dataframe where each line is separated, not the merged document. **The reason we are doing it that, as XCOMET has the max token length, if we pass the whole abstract, then it will not maybe able to produce the spans.** 

4. then it calls the model on it:

       In[9] model_output = model.predict(data, batch_size=8, gpus=1)

5. In [16] - working with the model tokenizer to tokenize a word and and reconstruct the word using the token ids assigned to each token.

**Main codes for getting spans for the community dataset start here:

In [22]: # create a json file with the results

# Save to a JSON file
    with open("output_all_spans_community.json", "w", encoding="utf-8") as f:

    json.dump(model_output.metadata.error_spans, f, ensure_ascii=False, indent=2)  # `indent` for readability
6. **tokenize the translation and postedition using xcomet's tokenizer** - this is to get the word level offsets for the dataset in order to facilitate word level mapping. This calculates the offsets of each word and stores the tokenized words for each sentence as well their offsets. 

7. **difflib helper function** - it just calculating the statistics of how many major, minor, critical spans are there using tokenized words by xcomet and saves the result at:
   **with open("all_entries_bug_fixed_with_opcodes_community.json", "w", encoding="utf-8") as f:**

8. **'Sentence level score' - heading:** - its just uses the regression head to provide sentence level scores and also later the codes also do the whole abstract level scores.

**dataset used for "confidence_score_solved_final.ipynb" and what codes and output files to look for:**

1. Input data: postedition_aligned.community.tsv
2. model: XCOMET already downloaded and the custom Xcomet class is instantiated with calling the checkpoint. In [5] and [6]
3. output data: "output_all_spans_community.json" - In [22], next In is In [7], these codes are getting the total count for major,minor etc spans from the whole dataset and used for initial statistics analysis how the dataset is distributed across spans. - [ ] do we need to do this for czech as well?
4. tokenized output for community data: **df.to_csv("postedition_aligned_with_tokenized_offsets_community.csv", index=False)** - [ ] look for it where it gets used, it should be used in calculating the word level mapping.
5. community dataset statistics of the proportion of spans: with open("all_entries_bug_fixed_with_opcodes_community.json", "w", encoding="utf-8") as f:
6. it produces all the json files needed for statistics, xcomet error spans:
   1. all_entries_bug_fixed_with_opcodes_community.json - final output file
   2. output_all_spans_community.json - contains xcomet spans final version
   3. postedition_aligned_with_tokenized_offsets_community.csv - this is the dataframe that adds tokenized words, offsets retrieved from xcomet to the original community dataset.
   4. relevant_words_in_pet_community_without_mapping.json - for each row in the dataframe the relevant words are extracted from the difflib opcodes where the words were replaced or deleted in the PE version from the original MT version, and this file contains entry for each row and the words that are relevant (replaced or deleted by the post editors in the MT sentence)
      
   what it does not produce or the codes were removed for generating the files:
   
   1. all_entries.json
   2. all_entries_bug_fixed.json
   3. all_entries_with_opcodes_translator.json
   4. postedition_aligned.final.translator.tsv
   5. postedition_aligned_with_tokenized_offsets_translator.csv
   6. relevant_words_in_pet_community.json
   7. relevant_words_in_pet_translator_without_mapping.json

### **So what needed to be done for English to Czech experiment:**

- [ ] 1. Load the dataset. first compute the abstract length of each document, then create the dataset in this way:
       - [ ] 1.1. only keep the sentences for a document that is part of it. then insert translation_id (you can start from 1 to 20) and postedit id (same, 1 to 20) and number each sentence in a document from 0 to doc_length -1.
       -
2. use the same model as english to frn experiment.
   from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/XCOMET-XL")
#model = load_from_checkpoint(model_path)
/storage/brno2/home/rahmang/envs/xcomet/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
Fetching 5 files: 100%|██████████| 5/5 [00:00<00:00, 22770.38it/s]

Then use the **CustomXCOMET** class and instantiate it using the checkpoint of the model. (In[6])

3. Then pass the dataset as list of dicts in this format:

        data = [
                 {"src": source,
                   "mt": mt,
                 }
                 {
                   }
                 ......
               ]
   where each dict is a line from the created document.

  - [ ] 3.1. then calculate the xcomet spans for each sentences and store them in "xcomet_spans.json"
  - [ ] 3.2. tokenize the sentences using xcomet tokenizer for later use to have word mapping logics. store it as "xcomet tokenized word and offsets_czech.json"
  - [ ] 3.3. do i also need to store difflib opcodes here?


### How relevant words are calculated:

those words were replaced, deleted from MT in the pe version, not the inserted ones.

#### So what does the **### confidence_score_solved_final.ipynb:** do:

1. It implements the custom xcomet class,
2. tokenize the sentences row wise in the df to get tokenize words and offsets for mt and pet sentences.
3. gets the xcomet spans
4. calculate the statistics of which words that were edited by the post editors actually fall into xcomet spans. for pedited words, it uses difflib sequence matcher and opcodes. and then it checks if the opcodes or changed words are inside the xcomet spans and then calculate the model performance in terms of tp, fp, tn, fn, - however, **the sequence mathcer works at word level.**
5. then it gets sentence level regression head scores and also for abstract level.
6. saves the relevant words found from difflib opcodes, df with tokenized words and offsets 
