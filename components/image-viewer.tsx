"use client"

import { useState } from "react"
import { Button } from "./ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs"
import { Maximize2, ZoomIn, ZoomOut, RotateCw, Sparkles } from "lucide-react"

interface ImageViewerProps {
  originalImage: string | null
  processedImage: string | null
}

export function ImageViewer({ originalImage, processedImage }: ImageViewerProps) {
  const [zoom, setZoom] = useState(1)
  const [rotation, setRotation] = useState(0)

  const handleZoomIn = () => {
    setZoom((prev) => Math.min(prev + 0.25, 3))
  }

  const handleZoomOut = () => {
    setZoom((prev) => Math.max(prev - 0.25, 0.5))
  }

  const handleResetView = () => {
    setZoom(1)
    setRotation(0)
  }

  const handleRotate = () => {
    setRotation((prev) => (prev + 90) % 360)
  }

  if (!originalImage) {
    return (
      <div className="w-full h-[500px] flex items-center justify-center bg-gray-900/30">
        <div className="text-center p-6 max-w-md">
          <div className="mx-auto w-20 h-20 rounded-full bg-gray-800/50 flex items-center justify-center mb-4">
            <Sparkles className="h-10 w-10 text-purple-500 opacity-70" />
          </div>
          <h3 className="text-xl font-medium text-gray-300 mb-2">No Image Selected</h3>
          <p className="text-gray-400">Upload an image or select a sample to begin exploring the cosmos</p>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full h-full">
      <Tabs defaultValue="original" className="w-full h-full flex flex-col">
        <div className="flex justify-between items-center p-4 border-b border-gray-800">
          <TabsList className="bg-gray-800/70">
            <TabsTrigger value="original" className="data-[state=active]:bg-purple-700 data-[state=active]:text-white">
              Original
            </TabsTrigger>
            <TabsTrigger
              value="processed"
              disabled={!processedImage}
              className="data-[state=active]:bg-purple-700 data-[state=active]:text-white"
            >
              Processed
            </TabsTrigger>
          </TabsList>

          <div className="flex gap-1">
            <Button
              variant="outline"
              size="icon"
              onClick={handleZoomIn}
              className="border-gray-700 bg-gray-800/50 hover:bg-gray-700"
            >
              <ZoomIn className="h-4 w-4 text-gray-300" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={handleZoomOut}
              className="border-gray-700 bg-gray-800/50 hover:bg-gray-700"
            >
              <ZoomOut className="h-4 w-4 text-gray-300" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={handleRotate}
              className="border-gray-700 bg-gray-800/50 hover:bg-gray-700"
            >
              <RotateCw className="h-4 w-4 text-gray-300" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={handleResetView}
              className="border-gray-700 bg-gray-800/50 hover:bg-gray-700"
            >
              <Maximize2 className="h-4 w-4 text-gray-300" />
            </Button>
          </div>
        </div>

        <div className="relative flex-grow overflow-hidden">
          <TabsContent value="original" className="m-0 h-full">
            <div className="w-full h-full flex items-center justify-center bg-black overflow-auto">
              <div
                style={{
                  transform: `scale(${zoom}) rotate(${rotation}deg)`,
                  transition: "transform 0.2s ease-in-out",
                }}
                className="relative"
              >
                {originalImage && (
                  <img
                    src={originalImage || "/placeholder.svg"}
                    alt="Original space image"
                    className="max-w-none"
                    style={{ objectFit: "contain" }}
                  />
                )}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="processed" className="m-0 h-full">
            <div className="w-full h-full flex items-center justify-center bg-black overflow-auto">
              {processedImage ? (
                <div
                  style={{
                    transform: `scale(${zoom}) rotate(${rotation}deg)`,
                    transition: "transform 0.2s ease-in-out",
                  }}
                  className="relative"
                >
                  <img
                    src={processedImage || "/placeholder.svg"}
                    alt="Processed space image with detected objects"
                    className="max-w-none"
                    style={{ objectFit: "contain" }}
                  />

                  {/* Overlay with detection markers (simulated) */}
                  <div className="absolute inset-0 pointer-events-none">
                    <div
                      className="absolute top-1/4 left-1/3 w-8 h-8 border-2 border-yellow-400 rounded-full"
                      style={{ boxShadow: "0 0 10px rgba(255, 255, 0, 0.7)" }}
                    ></div>
                    <div
                      className="absolute top-1/2 left-2/3 w-12 h-12 border-2 border-purple-500 rounded-full"
                      style={{ boxShadow: "0 0 10px rgba(147, 51, 234, 0.7)" }}
                    ></div>
                    <div
                      className="absolute top-3/4 left-1/4 w-10 h-10 border-2 border-blue-400 rounded-full"
                      style={{ boxShadow: "0 0 10px rgba(96, 165, 250, 0.7)" }}
                    ></div>
                  </div>
                </div>
              ) : (
                <div className="text-center p-6">
                  <p className="text-white">No processed image available</p>
                  <p className="text-sm text-gray-400 mt-2">Process the image to see results</p>
                </div>
              )}
            </div>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  )
}

