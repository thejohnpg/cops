import { type NextRequest, NextResponse } from "next/server"
import { spawn } from "child_process"
import { join } from "path"
import * as fs from "fs"

export async function POST(request: NextRequest) {
  try {
    // Processar o FormData diretamente
    const formData = await request.formData()
    const file = formData.get("file") as File

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    // Create temp directory if it doesn't exist
    const tempDir = join(process.cwd(), "temp")
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true })
    }

    // Create results directory if it doesn't exist
    const publicDir = join(process.cwd(), "public", "results")
    if (!fs.existsSync(publicDir)) {
      fs.mkdirSync(publicDir, { recursive: true })
    }

    // Convert the file to a Buffer
    const bytes = await file.arrayBuffer()
    const buffer = Buffer.from(bytes)

    // Generate a unique filename
    const timestamp = Date.now()
    const inputImagePath = join(tempDir, `input_${timestamp}.jpg`)
    const outputImagePath = join(tempDir, `output_${timestamp}.jpg`)
    const outputJsonPath = join(tempDir, `results_${timestamp}.json`)

    // Save the image to disk - explicitly convert Buffer to Uint8Array for TypeScript compatibility
    fs.writeFileSync(inputImagePath, new Uint8Array(buffer))

    // Call the Python script to process the image
    const pythonScriptPath = join(process.cwd(), "python", "celestial_detector.py")

    console.log("Executing Python script:", pythonScriptPath)
    console.log("Input image path:", inputImagePath)
    console.log("Output image path:", outputImagePath)
    console.log("Output JSON path:", outputJsonPath)

    // Execute the Python script
    const result = await new Promise<string>((resolve, reject) => {
      const pythonProcess = spawn("python", [pythonScriptPath, inputImagePath, outputImagePath, outputJsonPath])

      let output = ""
      let errorOutput = ""

      pythonProcess.stdout.on("data", (data) => {
        const dataStr = data.toString()
        console.log("Python output:", dataStr)
        output += dataStr
      })

      pythonProcess.stderr.on("data", (data) => {
        const dataStr = data.toString()
        console.error("Python Error:", dataStr)
        errorOutput += dataStr
      })

      pythonProcess.on("close", (code) => {
        console.log(`Python process exited with code ${code}`)
        if (code !== 0) {
          reject(new Error(`Python script exited with code ${code}: ${errorOutput}`))
        } else {
          resolve(output)
        }
      })
    })

    // Check if output files exist
    if (!fs.existsSync(outputImagePath)) {
      console.error("Output image file does not exist:", outputImagePath)
      return NextResponse.json({ error: "Failed to generate output image" }, { status: 500 })
    }

    if (!fs.existsSync(outputJsonPath)) {
      console.error("Output JSON file does not exist:", outputJsonPath)
      return NextResponse.json({ error: "Failed to generate detection results" }, { status: 500 })
    }

    // Read the results from the JSON file
    let results
    try {
      const jsonData = fs.readFileSync(outputJsonPath, "utf8")
      results = JSON.parse(jsonData)
    } catch (error) {
      console.error("Error reading results JSON:", error)
      return NextResponse.json({ error: "Failed to read detection results" }, { status: 500 })
    }

    // Create a public URL for the processed image
    const publicImageName = `processed_${timestamp}.jpg`
    const publicImagePath = join(publicDir, publicImageName)
    fs.copyFileSync(outputImagePath, publicImagePath)

    // Clean up temporary files
    try {
      fs.unlinkSync(inputImagePath)
      fs.unlinkSync(outputImagePath)
      fs.unlinkSync(outputJsonPath)
    } catch (error) {
      console.error("Error cleaning up temporary files:", error)
    }

    return NextResponse.json({
      success: true,
      processedImageUrl: `/results/${publicImageName}`,
      results,
    })
  } catch (error) {
    console.error("Error processing image:", error)
    // Retornar uma mensagem de erro mais detalhada
    const errorMessage = error instanceof Error ? error.message : "Unknown error occurred"
    return NextResponse.json(
      {
        error: "Failed to process image",
        details: errorMessage,
      },
      { status: 500 },
    )
  }
}

