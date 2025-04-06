"use client"

import { Badge } from "./ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table"
import { Star, Sparkles, CircleDot, AlertCircle } from "lucide-react"

interface CelestialObject {
  id: string
  type: "star" | "galaxy" | "nebula" | "unknown"
  confidence: number
  position: { x: number; y: number }
  size: number
  brightness: number
  color?: string
  name?: string
}

interface ResultsPanelProps {
  results: CelestialObject[] | null
}

export function ResultsPanel({ results }: ResultsPanelProps) {
  if (!results) {
    return (
      <div className="w-full h-[500px] flex items-center justify-center bg-gray-900/30">
        <div className="text-center p-6">
          <p className="text-gray-300">No detection results</p>
          <p className="text-sm text-gray-400 mt-2">Process an image to see celestial body detection results</p>
        </div>
      </div>
    )
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case "star":
        return <Star className="h-4 w-4 text-yellow-400" />
      case "galaxy":
        return <Sparkles className="h-4 w-4 text-purple-400" />
      case "nebula":
        return <CircleDot className="h-4 w-4 text-blue-400" />
      default:
        return <AlertCircle className="h-4 w-4 text-gray-400" />
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return "bg-green-900/30 text-green-400 border-green-700/50"
    if (confidence >= 0.7) return "bg-blue-900/30 text-blue-400 border-blue-700/50"
    if (confidence >= 0.5) return "bg-yellow-900/30 text-yellow-400 border-yellow-700/50"
    return "bg-red-900/30 text-red-400 border-red-700/50"
  }

  return (
    <div className="p-4 h-full overflow-auto">
      <div className="mb-4 flex flex-wrap gap-2">
        <Badge
          variant="outline"
          className="flex items-center gap-1 bg-yellow-900/20 border-yellow-700/30 text-yellow-400"
        >
          <Star className="h-3 w-3 text-yellow-400" /> Stars: {results.filter((r) => r.type === "star").length}
        </Badge>
        <Badge
          variant="outline"
          className="flex items-center gap-1 bg-purple-900/20 border-purple-700/30 text-purple-400"
        >
          <Sparkles className="h-3 w-3 text-purple-400" /> Galaxies: {results.filter((r) => r.type === "galaxy").length}
        </Badge>
        <Badge variant="outline" className="flex items-center gap-1 bg-blue-900/20 border-blue-700/30 text-blue-400">
          <CircleDot className="h-3 w-3 text-blue-400" /> Nebulae: {results.filter((r) => r.type === "nebula").length}
        </Badge>
        <Badge variant="outline" className="flex items-center gap-1 bg-gray-800/50 border-gray-700/30 text-gray-400">
          <AlertCircle className="h-3 w-3 text-gray-400" /> Unknown:{" "}
          {results.filter((r) => r.type === "unknown").length}
        </Badge>
      </div>

      <div className="rounded-md border border-gray-800 bg-gray-900/50 overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-gray-800/50 border-gray-800">
              <TableHead className="text-gray-300">Type</TableHead>
              <TableHead className="text-gray-300">Confidence</TableHead>
              <TableHead className="hidden md:table-cell text-gray-300">Size</TableHead>
              <TableHead className="hidden md:table-cell text-gray-300">Brightness</TableHead>
              <TableHead className="hidden lg:table-cell text-gray-300">Position</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {results.map((object) => (
              <TableRow key={object.id} className="hover:bg-gray-800/50 border-gray-800">
                <TableCell className="font-medium flex items-center gap-2 text-gray-300">
                  {getTypeIcon(object.type)}
                  <span className="capitalize">{object.name || object.type}</span>
                </TableCell>
                <TableCell>
                  <Badge variant="outline" className={getConfidenceColor(object.confidence)}>
                    {Math.round(object.confidence * 100)}%
                  </Badge>
                </TableCell>
                <TableCell className="hidden md:table-cell text-gray-400">{object.size.toFixed(2)}</TableCell>
                <TableCell className="hidden md:table-cell text-gray-400">{object.brightness.toFixed(2)}</TableCell>
                <TableCell className="hidden lg:table-cell text-gray-400">
                  x: {object.position.x.toFixed(0)}, y: {object.position.y.toFixed(0)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}

