"use client"

import { useState, useRef, useEffect } from "react"
import { Loader2 } from "lucide-react"

interface Position {
  x: number
  y: number
}

interface DetectedObject {
  type: string
  name: string | null
  confidence: number
  size: number
  magnitude: number
  position?: Position
  color?: string
  galaxy_type?: string
  nebula_type?: string
  distance?: number
  distance_unit?: string
  redshift?: number
  temperature?: number
  temperature_unit?: string
  mass?: number
  mass_unit?: string
  catalog?: string
}

interface ImageAnalysis {
  image_type: string
  object_type: string
  galaxy_name: string | null
  galaxy_confidence: number | null
  brightness: number
  contrast: number
  complexity: string
  edge_density: number
  dominant_color: string
  potential_redshift: boolean
  potential_blueshift: boolean
  color_distribution: {
    red: number
    green: number
    blue: number
  }
}

interface Summary {
  image_type: string
  total_objects: number
  object_counts: {
    stars: number
    galaxies: number
    nebulae: number
  }
  named_objects: {
    stars: string[]
    galaxies: string[]
    nebulae: string[]
  }
  brightness_level: string
  contrast_level: string
  complexity: string
  dominant_color: string
  spectral_shift?: string
  description: string
  notable_objects?: string
}

interface AnalysisResults {
  objects: DetectedObject[]
  summary: Summary
  image_analysis: ImageAnalysis
}

export default function Home() {
  const [originalImage, setOriginalImage] = useState<string | null>(null)
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  // Adicionar estado de loading
  const [isProcessing, setIsProcessing] = useState(false)
  const [activeTab, setActiveTab] = useState("viewer")
  const [results, setResults] = useState<AnalysisResults | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)

  // Estilo global para evitar problemas de hidratação
  useEffect(() => {
    const style = document.createElement("style")
    style.innerHTML = `
      * {
        box-sizing: border-box;
      }
      
      button {
        font-family: inherit;
      }
    `
    document.head.appendChild(style)

    return () => {
      document.head.removeChild(style)
    }
  }, [])

  const handleFileUpload = async (file: File) => {
    try {
      setIsProcessing(true)

      // Armazenar o arquivo localmente para processamento posterior
      setUploadedFile(file)

      // Criar uma URL local para visualização
      const localUrl = URL.createObjectURL(file)
      setOriginalImage(localUrl)

      // Reset previous results
      setProcessedImage(null)
      setResults(null)

      setIsProcessing(false)
    } catch (error) {
      console.error("Error handling file upload:", error)
      setIsProcessing(false)
    }
  }

  const handleSampleImage = async (samplePath: string) => {
    try {
      setIsProcessing(true)

      // Fetch the sample image and convert to File object
      const response = await fetch(samplePath)
      const blob = await response.blob()
      const file = new File([blob], samplePath.split("/").pop() || "sample.jpg", { type: blob.type })

      // Armazenar o arquivo localmente para processamento posterior
      setUploadedFile(file)

      // Criar uma URL local para visualização
      const localUrl = URL.createObjectURL(file)
      setOriginalImage(localUrl)

      // Reset previous results
      setProcessedImage(null)
      setResults(null)

      setIsProcessing(false)
    } catch (error) {
      console.error("Error loading sample image:", error)
      setIsProcessing(false)
    }
  }

  // Modificar a função processImage para mostrar o loading
  const processImage = async () => {
    if (!uploadedFile) return

    setIsProcessing(true)

    try {
      // Criar um FormData para enviar o arquivo diretamente para o servidor
      const formData = new FormData()
      formData.append("file", uploadedFile)

      // Enviar o arquivo para processamento no servidor
      const response = await fetch("/api/process-image", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Error processing image: ${response.statusText}`)
      }

      const data = await response.json()

      // Set the processed image with annotations
      setProcessedImage(data.processedImageUrl)

      // Set the detection results
      setResults(data.results)

      // Mudar automaticamente para a aba de resultados
      setActiveTab("results")
    } catch (error) {
      console.error("Error processing image:", error)
      alert("Error processing image. Please try again.")
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <main
      style={{
        minHeight: "100vh",
        background: "linear-gradient(to bottom, #000000, #0f0f1e, #1a1a40)",
        color: "white",
        padding: "20px",
      }}
    >
      {/* Hero Section */}
      <div style={{ maxWidth: "1200px", margin: "0 auto" }}>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-start",
            marginBottom: "30px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center" }}>
            <div
              style={{
                marginRight: "15px",
                background: "#5d3fd3",
                borderRadius: "50%",
                width: "40px",
                height: "40px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              🚀
            </div>
            <h1
              style={{
                fontSize: "2.5rem",
                fontWeight: "bold",
                background: "linear-gradient(to right, #a78bfa, #ec4899, #f59e0b)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                margin: 0,
              }}
            >
              Celestial Explorer
            </h1>
          </div>
          <p style={{ color: "#d1d5db", maxWidth: "600px", marginTop: "10px" }}>
            Discover the universe through advanced image processing. Upload space images to detect and identify
            celestial bodies.
          </p>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
            gap: "24px",
          }}
        >
          {/* Upload Panel */}
          <div
            style={{
              background: "rgba(17, 24, 39, 0.7)",
              borderRadius: "12px",
              border: "1px solid rgba(109, 40, 217, 0.3)",
              overflow: "hidden",
              boxShadow: "0 4px 20px rgba(0, 0, 0, 0.5)",
            }}
          >
            <div
              style={{
                padding: "20px",
                borderBottom: "1px solid rgba(109, 40, 217, 0.2)",
                display: "flex",
                alignItems: "center",
              }}
            >
              <span style={{ marginRight: "10px" }}>📷</span>
              <h2 style={{ fontSize: "1.25rem", fontWeight: "600", margin: 0 }}>Image Upload</h2>
            </div>

            <div style={{ padding: "20px" }}>
              {/* Upload Area */}
              <div
                style={{
                  border: "2px dashed #4b5563",
                  borderRadius: "12px",
                  padding: "24px",
                  textAlign: "center",
                  cursor: "pointer",
                  transition: "all 0.3s ease",
                }}
                onClick={() => fileInputRef.current?.click()}
              >
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "12px" }}>
                  <div
                    style={{
                      height: "64px",
                      width: "64px",
                      borderRadius: "50%",
                      background: "#1f2937",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: "24px",
                    }}
                  >
                    📤
                  </div>
                  <p style={{ fontSize: "0.875rem", color: "#d1d5db", margin: 0 }}>
                    Drag and drop an image, or click to browse
                  </p>
                  <p style={{ fontSize: "0.75rem", color: "#6b7280", margin: 0 }}>Supports: JPG, PNG, TIFF</p>
                </div>
                <input
                  ref={fileInputRef}
                  id="file-input"
                  type="file"
                  accept="image/*"
                  style={{ display: "none" }}
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      handleFileUpload(e.target.files[0])
                    }
                  }}
                />
              </div>

              {/* Process Button */}
              <div style={{ marginTop: "20px" }}>
                {/* Modificar o botão de processamento para mostrar o loading */}
                <button
                  onClick={processImage}
                  disabled={!uploadedFile || isProcessing}
                  style={{
                    width: "100%",
                    padding: "10px",
                    background: "linear-gradient(to right, #8b5cf6, #d946ef)",
                    border: "none",
                    borderRadius: "6px",
                    color: "white",
                    fontWeight: "500",
                    cursor: uploadedFile && !isProcessing ? "pointer" : "not-allowed",
                    opacity: uploadedFile && !isProcessing ? "1" : "0.5",
                    transition: "all 0.3s ease",
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    gap: "8px",
                  }}
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    "Analyze Image"
                  )}
                </button>
              </div>

              {/* Sample Images */}
              <div style={{ marginTop: "24px" }}>
                <h3 style={{ fontSize: "0.875rem", fontWeight: "500", color: "#d1d5db", marginBottom: "12px" }}>
                  Sample Images
                </h3>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
                  <button
                    style={{
                      padding: "8px",
                      background: "rgba(109, 40, 217, 0.2)",
                      border: "1px solid rgba(109, 40, 217, 0.3)",
                      borderRadius: "6px",
                      color: "#c4b5fd",
                      cursor: "pointer",
                    }}
                    onClick={() => handleSampleImage("/samples/hubble-deep-field.jpg")}
                  >
                    Hubble Deep Field
                  </button>
                  <button
                    style={{
                      padding: "8px",
                      background: "rgba(109, 40, 217, 0.2)",
                      border: "1px solid rgba(109, 40, 217, 0.3)",
                      borderRadius: "6px",
                      color: "#c4b5fd",
                      cursor: "pointer",
                    }}
                    onClick={() => handleSampleImage("/samples/andromeda.jpg")}
                  >
                    Andromeda Galaxy
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Viewer Panel */}
          <div
            style={{
              background: "rgba(17, 24, 39, 0.7)",
              borderRadius: "12px",
              border: "1px solid rgba(109, 40, 217, 0.3)",
              overflow: "hidden",
              boxShadow: "0 4px 20px rgba(0, 0, 0, 0.5)",
              display: "flex",
              flexDirection: "column",
              minHeight: "500px",
            }}
          >
            <div
              style={{
                padding: "16px",
                borderBottom: "1px solid rgba(109, 40, 217, 0.2)",
                display: "flex",
                justifyContent: "space-between",
              }}
            >
              <div
                style={{
                  display: "flex",
                  background: "#1f2937",
                  borderRadius: "6px",
                  padding: "4px",
                }}
              >
                <button
                  style={{
                    padding: "6px 12px",
                    background: activeTab === "viewer" ? "#7c3aed" : "transparent",
                    border: "none",
                    borderRadius: "4px",
                    color: activeTab === "viewer" ? "white" : "#9ca3af",
                    cursor: "pointer",
                  }}
                  onClick={() => setActiveTab("viewer")}
                >
                  Image Viewer
                </button>
                <button
                  style={{
                    padding: "6px 12px",
                    background: activeTab === "results" ? "#7c3aed" : "transparent",
                    border: "none",
                    borderRadius: "4px",
                    color: activeTab === "results" ? "white" : "#9ca3af",
                    cursor: results ? "pointer" : "not-allowed",
                    opacity: results ? "1" : "0.5",
                  }}
                  onClick={() => results && setActiveTab("results")}
                  disabled={!results}
                >
                  Analysis Results
                </button>
              </div>
            </div>

            <div style={{ flex: "1", overflow: "hidden" }}>
              {activeTab === "viewer" ? (
                <div style={{ width: "100%", height: "100%" }}>
                  {!originalImage ? (
                    <div
                      style={{
                        width: "100%",
                        height: "100%",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        background: "rgba(17, 24, 39, 0.3)",
                      }}
                    >
                      <div style={{ textAlign: "center", padding: "24px", maxWidth: "400px" }}>
                        <div
                          style={{
                            width: "80px",
                            height: "80px",
                            borderRadius: "50%",
                            background: "#1f2937",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: "32px",
                            margin: "0 auto 16px auto",
                          }}
                        >
                          ✨
                        </div>
                        <h3 style={{ fontSize: "1.25rem", fontWeight: "500", color: "#d1d5db", marginBottom: "8px" }}>
                          No Image Selected
                        </h3>
                        <p style={{ color: "#9ca3af" }}>
                          Upload an image or select a sample to begin exploring the cosmos
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div
                      style={{
                        width: "100%",
                        height: "100%",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        background: "black",
                        overflow: "auto",
                      }}
                    >
                      <img
                        src={processedImage || originalImage}
                        alt="Space image"
                        style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
                      />
                    </div>
                  )}
                </div>
              ) : (
                // Adicionar indicador de loading na área de resultados
                <div style={{ padding: "16px", height: "100%", overflowY: "auto" }}>
                  {isProcessing ? (
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                        height: "100%",
                        gap: "16px",
                      }}
                    >
                      <Loader2 className="h-12 w-12 text-purple-500 animate-spin" />
                      <div>
                        <p style={{ color: "#d1d5db", fontSize: "1.1rem", fontWeight: "500" }}>Analyzing image...</p>
                        <p style={{ color: "#9ca3af", fontSize: "0.875rem", marginTop: "4px" }}>
                          This may take a moment as we process your image with our AI models
                        </p>
                      </div>
                    </div>
                  ) : results && results.summary ? (
                    // Conteúdo existente dos resultados
                    <>
                      {/* Image Summary */}
                      <div
                        style={{
                          padding: "16px",
                          background: "rgba(31, 41, 55, 0.5)",
                          borderRadius: "8px",
                          marginBottom: "16px",
                        }}
                      >
                        <h3
                          style={{
                            fontSize: "1.1rem",
                            fontWeight: "600",
                            color: "#e5e7eb",
                            marginTop: 0,
                            marginBottom: "12px",
                          }}
                        >
                          Image Analysis
                        </h3>
                        <p style={{ color: "#d1d5db", margin: "0 0 8px 0", lineHeight: "1.5" }}>
                          {results.summary.description}
                        </p>
                        {results.summary.notable_objects && (
                          <p style={{ color: "#d1d5db", margin: "8px 0", lineHeight: "1.5" }}>
                            {results.summary.notable_objects}
                          </p>
                        )}
                        <div
                          style={{
                            display: "grid",
                            gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
                            gap: "8px",
                            marginTop: "12px",
                          }}
                        >
                          <div style={{ background: "rgba(17, 24, 39, 0.5)", padding: "8px", borderRadius: "6px" }}>
                            <p style={{ fontSize: "0.75rem", color: "#9ca3af", margin: "0 0 4px 0" }}>Image Type</p>
                            <p style={{ fontSize: "0.875rem", color: "#e5e7eb", margin: 0, fontWeight: "500" }}>
                              {results.summary.image_type}
                            </p>
                          </div>
                          <div style={{ background: "rgba(17, 24, 39, 0.5)", padding: "8px", borderRadius: "6px" }}>
                            <p style={{ fontSize: "0.75rem", color: "#9ca3af", margin: "0 0 4px 0" }}>Brightness</p>
                            <p style={{ fontSize: "0.875rem", color: "#e5e7eb", margin: 0, fontWeight: "500" }}>
                              {results.summary.brightness_level}
                            </p>
                          </div>
                          <div style={{ background: "rgba(17, 24, 39, 0.5)", padding: "8px", borderRadius: "6px" }}>
                            <p style={{ fontSize: "0.75rem", color: "#9ca3af", margin: "0 0 4px 0" }}>Contrast</p>
                            <p style={{ fontSize: "0.875rem", color: "#e5e7eb", margin: 0, fontWeight: "500" }}>
                              {results.summary.contrast_level}
                            </p>
                          </div>
                          <div style={{ background: "rgba(17, 24, 39, 0.5)", padding: "8px", borderRadius: "6px" }}>
                            <p style={{ fontSize: "0.75rem", color: "#9ca3af", margin: "0 0 4px 0" }}>Complexity</p>
                            <p style={{ fontSize: "0.875rem", color: "#e5e7eb", margin: 0, fontWeight: "500" }}>
                              {results.summary.complexity}
                            </p>
                          </div>
                        </div>

                        {results.summary.spectral_shift && (
                          <div
                            style={{
                              background: "rgba(17, 24, 39, 0.5)",
                              padding: "8px",
                              borderRadius: "6px",
                              marginTop: "8px",
                            }}
                          >
                            <p style={{ fontSize: "0.75rem", color: "#9ca3af", margin: "0 0 4px 0" }}>
                              Spectral Analysis
                            </p>
                            <p style={{ fontSize: "0.875rem", color: "#e5e7eb", margin: 0, fontWeight: "500" }}>
                              {results.summary.spectral_shift}
                            </p>
                          </div>
                        )}
                      </div>

                      {/* Object Counts */}
                      <div
                        style={{
                          display: "flex",
                          flexWrap: "wrap",
                          gap: "8px",
                          marginBottom: "16px",
                        }}
                      >
                        <span
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "4px",
                            padding: "4px 10px",
                            background: "rgba(234, 179, 8, 0.2)",
                            border: "1px solid rgba(234, 179, 8, 0.3)",
                            borderRadius: "9999px",
                            fontSize: "0.75rem",
                            color: "#fcd34d",
                          }}
                        >
                          ⭐ Stars: {results.summary.object_counts.stars}
                        </span>
                        <span
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "4px",
                            padding: "4px 10px",
                            background: "rgba(139, 92, 246, 0.2)",
                            border: "1px solid rgba(139, 92, 246, 0.3)",
                            borderRadius: "9999px",
                            fontSize: "0.75rem",
                            color: "#c4b5fd",
                          }}
                        >
                          🌌 Galaxies: {results.summary.object_counts.galaxies}
                        </span>
                        <span
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "4px",
                            padding: "4px 10px",
                            background: "rgba(59, 130, 246, 0.2)",
                            border: "1px solid rgba(59, 130, 246, 0.3)",
                            borderRadius: "9999px",
                            fontSize: "0.75rem",
                            color: "#93c5fd",
                          }}
                        >
                          🌠 Nebulae: {results.summary.object_counts.nebulae}
                        </span>
                      </div>

                      {/* Detected Objects Table */}
                      <div
                        style={{
                          border: "1px solid #374151",
                          borderRadius: "6px",
                          overflow: "hidden",
                          marginBottom: "16px",
                        }}
                      >
                        <div style={{ padding: "12px 16px", background: "rgba(31, 41, 55, 0.7)" }}>
                          <h3 style={{ fontSize: "1rem", fontWeight: "600", color: "#e5e7eb", margin: 0 }}>
                            Detected Celestial Objects
                          </h3>
                        </div>
                        <table style={{ width: "100%", borderCollapse: "collapse" }}>
                          <thead>
                            <tr style={{ borderBottom: "1px solid #374151", background: "rgba(31, 41, 55, 0.5)" }}>
                              <th style={{ padding: "12px 16px", textAlign: "left", color: "#d1d5db" }}>Object</th>
                              <th style={{ padding: "12px 16px", textAlign: "left", color: "#d1d5db" }}>Type</th>
                              <th style={{ padding: "12px 16px", textAlign: "left", color: "#d1d5db" }}>Confidence</th>
                              <th style={{ padding: "12px 16px", textAlign: "left", color: "#d1d5db" }}>Size (px)</th>
                              <th style={{ padding: "12px 16px", textAlign: "left", color: "#d1d5db" }}>Magnitude</th>
                            </tr>
                          </thead>
                          <tbody>
                            {results.objects.map((object, index) => (
                              <tr key={index} style={{ borderBottom: "1px solid #374151" }}>
                                <td style={{ padding: "12px 16px", color: "#d1d5db" }}>
                                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                    <span
                                      style={{
                                        color:
                                          object.type === "star"
                                            ? "#fcd34d"
                                            : object.type === "galaxy"
                                              ? "#c4b5fd"
                                              : "#93c5fd",
                                      }}
                                    >
                                      {object.type === "star" ? "⭐" : object.type === "galaxy" ? "🌌" : "🌠"}
                                    </span>
                                    <span>{object.name || "Unnamed"}</span>
                                    {object.catalog && (
                                      <span style={{ fontSize: "0.75rem", color: "#9ca3af" }}>({object.catalog})</span>
                                    )}
                                  </div>
                                </td>
                                <td style={{ padding: "12px 16px", color: "#d1d5db" }}>
                                  {object.type === "star"
                                    ? object.color
                                      ? `${object.color} star`
                                      : "Star"
                                    : object.type === "galaxy"
                                      ? object.galaxy_type
                                        ? `${object.galaxy_type} galaxy`
                                        : "Galaxy"
                                      : object.nebula_type
                                        ? `${object.nebula_type} nebula`
                                        : "Nebula"}
                                </td>
                                <td style={{ padding: "12px 16px" }}>
                                  <span
                                    style={{
                                      padding: "2px 8px",
                                      background:
                                        object.confidence >= 0.9
                                          ? "rgba(22, 163, 74, 0.2)"
                                          : object.confidence >= 0.7
                                            ? "rgba(59, 130, 246, 0.2)"
                                            : "rgba(234, 179, 8, 0.2)",
                                      border:
                                        object.confidence >= 0.9
                                          ? "1px solid rgba(22, 163, 74, 0.3)"
                                          : object.confidence >= 0.7
                                            ? "1px solid rgba(59, 130, 246, 0.3)"
                                            : "1px solid rgba(234, 179, 8, 0.3)",
                                      borderRadius: "9999px",
                                      fontSize: "0.75rem",
                                      color:
                                        object.confidence >= 0.9
                                          ? "#86efac"
                                          : object.confidence >= 0.7
                                            ? "#93c5fd"
                                            : "#fcd34d",
                                    }}
                                  >
                                    {Math.round(object.confidence * 100)}%
                                  </span>
                                </td>
                                <td style={{ padding: "12px 16px", color: "#9ca3af" }}>{object.size.toFixed(2)}</td>
                                <td style={{ padding: "12px 16px", color: "#9ca3af" }}>
                                  {object.magnitude !== null && object.magnitude !== undefined
                                    ? object.magnitude.toFixed(2)
                                    : "N/A"}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>

                      {/* Scientific Data Table */}
                      <div
                        style={{
                          border: "1px solid #374151",
                          borderRadius: "6px",
                          overflow: "hidden",
                          marginBottom: "16px",
                        }}
                      >
                        <div style={{ padding: "12px 16px", background: "rgba(31, 41, 55, 0.7)" }}>
                          <h3 style={{ fontSize: "1rem", fontWeight: "600", color: "#e5e7eb", margin: 0 }}>
                            Scientific Data
                          </h3>
                        </div>
                        <table style={{ width: "100%", borderCollapse: "collapse" }}>
                          <thead>
                            <tr style={{ borderBottom: "1px solid #374151", background: "rgba(31, 41, 55, 0.5)" }}>
                              <th style={{ padding: "12px 16px", textAlign: "left", color: "#d1d5db" }}>Object</th>
                              <th style={{ padding: "12px 16px", textAlign: "left", color: "#d1d5db" }}>Distance</th>
                              <th style={{ padding: "12px 16px", textAlign: "left", color: "#d1d5db" }}>
                                Red/Blueshift
                              </th>
                              <th style={{ padding: "12px 16px", textAlign: "left", color: "#d1d5db" }}>Mass</th>
                              <th style={{ padding: "12px 16px", textAlign: "left", color: "#d1d5db" }}>Temperature</th>
                            </tr>
                          </thead>
                          <tbody>
                            {results.objects
                              .filter((obj) => obj.distance || obj.redshift || obj.mass || obj.temperature)
                              .map((object, index) => (
                                <tr key={index} style={{ borderBottom: "1px solid #374151" }}>
                                  <td style={{ padding: "12px 16px", color: "#d1d5db" }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                      <span
                                        style={{
                                          color:
                                            object.type === "star"
                                              ? "#fcd34d"
                                              : object.type === "galaxy"
                                                ? "#c4b5fd"
                                                : "#93c5fd",
                                        }}
                                      >
                                        {object.type === "star" ? "⭐" : object.type === "galaxy" ? "🌌" : "🌠"}
                                      </span>
                                      <span>{object.name || "Unnamed"}</span>
                                    </div>
                                  </td>
                                  <td style={{ padding: "12px 16px", color: "#9ca3af" }}>
                                    {object.distance ? `${object.distance} ${object.distance_unit || ""}` : "N/A"}
                                  </td>
                                  <td style={{ padding: "12px 16px", color: "#9ca3af" }}>
                                    {object.redshift !== undefined
                                      ? object.redshift > 0
                                        ? `Redshift: ${object.redshift.toFixed(6)}`
                                        : object.redshift < 0
                                          ? `Blueshift: ${Math.abs(object.redshift).toFixed(6)}`
                                          : "No shift"
                                      : "N/A"}
                                  </td>
                                  <td style={{ padding: "12px 16px", color: "#9ca3af" }}>
                                    {object.mass ? `${object.mass.toExponential(2)} ${object.mass_unit || ""}` : "N/A"}
                                  </td>
                                  <td style={{ padding: "12px 16px", color: "#9ca3af" }}>
                                    {object.temperature
                                      ? `${object.temperature} ${object.temperature_unit || ""}`
                                      : "N/A"}
                                  </td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                      </div>

                      {/* Explanation */}
                      <div style={{ padding: "16px", background: "rgba(31, 41, 55, 0.5)", borderRadius: "8px" }}>
                        <h3
                          style={{
                            fontSize: "1rem",
                            fontWeight: "600",
                            color: "#e5e7eb",
                            marginTop: 0,
                            marginBottom: "12px",
                          }}
                        >
                          Understanding These Results
                        </h3>
                        <div style={{ color: "#d1d5db", fontSize: "0.875rem", lineHeight: "1.5" }}>
                          <p style={{ margin: "0 0 8px 0" }}>
                            <strong>Object Type:</strong> Identifies the celestial object as a star, galaxy, or nebula,
                            along with specific characteristics.
                          </p>
                          <p style={{ margin: "0 0 8px 0" }}>
                            <strong>Confidence:</strong> Indicates how certain our system is about the classification of
                            the object.
                          </p>
                          <p style={{ margin: "0 0 8px 0" }}>
                            <strong>Size:</strong> Represents the relative diameter of the object in the image (in
                            pixels).
                          </p>
                          <p style={{ margin: "0 0 8px 0" }}>
                            <strong>Magnitude:</strong> Indicates the brightness of the object. Lower values represent
                            brighter objects (astronomical magnitude scale).
                          </p>
                          <p style={{ margin: "0 0 8px 0" }}>
                            <strong>Distance:</strong> How far the object is from Earth, measured in light-years for
                            stars and million light-years for galaxies.
                          </p>
                          <p style={{ margin: "0 0 8px 0" }}>
                            <strong>Red/Blueshift:</strong> Indicates whether the object is moving away from us
                            (redshift) or toward us (blueshift).
                          </p>
                          <p style={{ margin: "8px 0 0 0" }}>
                            The processed image shows the detected objects with labels and scientific data. Named
                            objects are identified based on their characteristics and position in the image.
                          </p>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div style={{ textAlign: "center", padding: "24px" }}>
                      <p style={{ color: "#d1d5db" }}>No analysis results available</p>
                      <p style={{ color: "#9ca3af", fontSize: "0.875rem", marginTop: "8px" }}>
                        Process an image to see celestial body detection results
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}

