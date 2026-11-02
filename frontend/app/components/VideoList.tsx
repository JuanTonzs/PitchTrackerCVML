"use client";

import { Search, X } from "lucide-react";
import { Video } from "../types";
import React, { useState } from 'react';


interface SearchableVideoListProps {
    videos: Video[];
    searchText: string;
    onSearchChange: (text: string) => void;
}

export function SearchableVideoList({
    videos,
    searchText,
    onSearchChange,
}: SearchableVideoListProps) {
    
    const[selectedVideo, setSelectedVideo] = useState<Video | null>(null);
    
    const foundVideos = videos.filter((video) => video.title.toLowerCase().includes(searchText.toLowerCase()) ||
    video.description.toLowerCase().includes(searchText.toLowerCase())
    );


    return (
        <div className="max-w-2xl mx-auto p-6 bg-white shadow-xl rounded-2xl my-8 font-sans">
            <h1 className="text-3xl font-bold text-gray-900">Tracked Baseball Videos</h1>
            <div className="relative mb-6">
                <span className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none text-gray-400">
                    <Search className="w-5 h-5" />
                </span>
                <input 
                    type="text"
                    value={searchText}
                    onChange={(e) => onSearchChange(e.target.value)}
                    placeholder="Search"
                    className="w-full pl-10 pr-4 py-3 bg-gray-100 rounded-xl border-none focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-700 placeholder-gray-400"
                />
            </div>

            {/* Literally checks if more than one video make it plural lol */}
            <h2 className="text-xl font-bold text-gray-900 mb-4">
                {foundVideos.length} {foundVideos.length === 1 ? 'Video' : 'Videos'}                
            </h2>

            {foundVideos.length === 0 ? (
                <p className="text-gray-400 text-center py-8" >No matches for "{searchText}"</p>
            ) : (
                <div className="space-y-4"> 
                    {foundVideos.map((video) => (
                        <div
                            key={video.id}
                            onClick={() => setSelectedVideo(video)}
                            className="flex items-center justify-between p-3 bg-white border border-gray-100 rounded-xl shadow-sm hover:shadow-md transition-shadow"
                        >
                            <div className="flex items-center space-x-4">
                                {/* Video Player/Thumbnail Preview */}
                                <video
                                    src={video.videoURL}

                                    className="w-32 h-20 object-cover rounded-lg bg-black"
                                />
                                <div>
                                    <h3 className="font-semibold text-gray-900">{video.title}</h3>
                                    <p className="text-sm text-gray-500 line-clamp-1">{video.description}</p>
                                </div>
                            </div>
                            
                        </div>
                    ))}
                </div>
            )}


            {selectedVideo && (
                <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"> 
                    <div className="bg-white rounded-2xl max-w-2xl w-full p-6 relative shadow-2xl">
                        <button onClick={() => setSelectedVideo(null)}
                            className="absolute top-4 right-4 text-gray-400 hover:text-gray-700 p-1"

                        >
                            <X className="w-6 h-6" />
                        </button>
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">{selectedVideo.title}</h3>
                        <p className="text-gray-500 mb-4">{selectedVideo.description}</p>

                        <video 
                            src={selectedVideo.videoURL}
                            controls
                            autoPlay
                            className="w-full max-h-[400px] rounded-xl bg-black"
                        />
                    </div>
                </div>

            )}
        </div>

        
    );
}