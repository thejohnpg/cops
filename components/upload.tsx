"use client"

import type React from "react"

import { useState, useRef } from "react"
import { UploadCloud, ImageIcon } from "lucide-react"

interface UploadProps {
  onImageUploaded: (imageUrl: string) => void
}

export function Upload({ onImageUploaded }: UploadProps) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = async (file: File) => {
    // In a real app, you would upload the file to the server
    // For this demo, we'll just create a local URL
    const imageUrl = URL.createObjectURL(file)
    onImageUploaded(imageUrl)
  }

  return (
    <div
      className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-300 ${
        isDragging
          ? "border-purple-500 bg-purple-900/20"
          : "border-gray-700 hover:border-purple-700 hover:bg-purple-900/10"
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
    >
      <input type="file" ref={fileInputRef} onChange={handleFileInput} accept="image/*" className="hidden" />

      <div className="flex flex-col items-center justify-center gap-3">
        {isDragging ? (
          <div className="h-16 w-16 rounded-full bg-purple-900/30 flex items-center justify-center">
            <ImageIcon className="h-8 w-8 text-purple-400" />
          </div>
        ) : (
          <div className="h-16 w-16 rounded-full bg-gray-800/50 flex items-center justify-center">
            <UploadCloud className="h-8 w-8 text-gray-400" />
          </div>
        )}
        <p className="text-sm text-gray-300">Drag and drop an image, or click to browse</p>
        <p className="text-xs text-gray-500">Supports: JPG, PNG, TIFF</p>
      </div>
    </div>
  )
}

