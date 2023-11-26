#!/bin/bash

# Define the source directory where your documents are located
source_dir="/Users/maguo/Desktop/Psychological-Therapeutic-Chatbot/data/transcript1"

# Define the destination directory where you want to move the documents
destination_dir="/Users/maguo/Desktop/Psychological-Therapeutic-Chatbot/data/transcript"

# Define the text file containing the document names (one per line)
document_list_file="/Users/maguo/Desktop/Psychological-Therapeutic-Chatbot/data/volumn1_transcripts_filename.txt"

# Check if the source directory exists
if [ ! -d "$source_dir" ]; then
    echo "Source directory does not exist: $source_dir"
    exit 1
fi

# Check if the destination directory exists; create it if not
if [ ! -d "$destination_dir" ]; then
    mkdir -p "$destination_dir"
fi

# Check if the document list file exists
if [ ! -f "$document_list_file" ]; then
    echo "Document list file does not exist: $document_list_file"
    exit 1
fi

# Loop through each document name in the list and move it to the destination directory
while IFS= read -r document_name; do
    source_path="$source_dir/$document_name"
    destination_path="$destination_dir/$document_name"

    if [ -e "$source_path" ]; then
        mv "$source_path" "$destination_path"
        echo "Moved $document_name to $destination_dir"
    else
        echo "Document not found: $document_name"
    fi
done < "$document_list_file"
