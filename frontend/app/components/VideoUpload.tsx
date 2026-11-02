"use client";

import React, { useState } from "react";
import { Video } from "../types";
import { Upload } from "lucide-react";

interface VideoUploadProps {
    onVideoUpload: (newVideo: Video) => void;
}

export function VideoUpload({ onVideoUpload }: VideoUploadProps) {
    const[title, setTitle] = useState("");
    const[description, setDescription] = useState("");
    const[file, setFile] = useState<File | null>(null)
    

    // TODO: At some point this will be sent back to backend and bring back new video with tracked data 
    const handleSubmit = (e: React.SubmitEvent<HTMLFormElement>) => {
        e.preventDefault();
        if(!file || !title) return;

        const videoURL = URL.createObjectURL(file);

        // Adds the metadata to the video
        const newVideo: Video = {
            id: Date.now().toString(),
            title,
            description,
            videoURL,
            createdAt: new Date().toLocaleString(),
        };

        onVideoUpload(newVideo);

        setTitle('');
        setDescription('');
        setFile(null);
    };


    return (
        <div className="max-w-2xl mx-auto p-6 bg-white shadow-xl rounded-2xl my-8">
            <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                <Upload className="w-5 h-5 text-indigo-600" />Upload a Video!
            </h2>

            <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Video Title</label>
                    <input 
                        type="text"
                        required
                        value={title}
                        onChange={(e) => setTitle(e.target.value)}
                        placeholder="Add Title"
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                    <input 
                        type="text"
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        placeholder="Add Description"
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                    /> 
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Video File</label>
                    <input 
                        type="file"
                        accept="video/*"
                        required
                        onChange={(e) => setFile(e.target.files?.[0] || null)}
                        className="w-full py-3 bg-indigo-600 text-white font-semibold rounded-xl hover:bg-indigo-700 transition-colors shadow-md"
                    />
                </div>

                <button
                    type="submit"
                    className="w-full py-3 bg-indigo-600 text-white font-semibold rounded-xl hover:bg-indigo-700 transition-colors shadow-md"
                >
                    Add Video
                </button>
            </form>
        </div>
    );
}