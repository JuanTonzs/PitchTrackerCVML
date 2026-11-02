"use client";

import { useState } from "react";
import { Video } from "./types";
import { VideoUpload } from "./components/VideoUpload";
import { SearchableVideoList } from "./components/VideoList";

export default function Home() {
  const [searchText, setSearchText] = useState("");

  // RANDOM PLACEHOLDER VIDEOS
  const [videos, setVideos] = useState<Video[]>([
    {
      id: '1',
      title: 'React: The Documentary',
      description: 'The origin story of React',
      videoURL: 'https://www.w3schools.com/html/mov_bbb.mp4',
      createdAt: '2023',
    },
    {
      id: '2',
      title: 'Rethinking Best Practices',
      description: 'Pete Hunt (2013)',
      videoURL: 'https://www.w3schools.com/html/mov_bbb.mp4',
      createdAt: '2013',
    },
  ]);

  const handleVideoUpload = (newVideo: Video) => {
    setVideos((prevVideos) => [newVideo, ...prevVideos])
  }

  return (
    <div className="min-h-screen bg-gray-50 py-10 px-4 font-sans">
      <div className="max-w-2xl mx-auto space-y-6">
        <VideoUpload onVideoUpload={handleVideoUpload} />

        <SearchableVideoList 
          videos={videos}
          searchText={searchText}
          onSearchChange={setSearchText}
        />
      </div>
    </div>
  );
}
