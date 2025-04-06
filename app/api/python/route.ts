import { type NextRequest, NextResponse } from "next/server"
import { spawn } from "child_process"
import { join } from "path"

export async function POST(request: NextRequest) {
  try {
    const { script, args = [] } = await request.json()

    if (!script) {
      return NextResponse.json({ error: "No script provided" }, { status: 400 })
    }

    // Execute the Python script
    const result = await executePythonScript(script, args)

    return NextResponse.json({
      success: true,
      result,
    })
  } catch (error) {
    console.error("Error executing Python script:", error)
    return NextResponse.json({ error: "Failed to execute Python script" }, { status: 500 })
  }
}

async function executePythonScript(scriptPath: string, args: string[] = []): Promise<string> {
  return new Promise((resolve, reject) => {
    // Get the absolute path to the Python script
    const absoluteScriptPath = join(process.cwd(), scriptPath)

    // Spawn a Python process
    const pythonProcess = spawn("python", [absoluteScriptPath, ...args])

    let output = ""
    let errorOutput = ""

    // Collect stdout data
    pythonProcess.stdout.on("data", (data) => {
      output += data.toString()
    })

    // Collect stderr data
    pythonProcess.stderr.on("data", (data) => {
      errorOutput += data.toString()
    })

    // Handle process completion
    pythonProcess.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Python script exited with code ${code}: ${errorOutput}`))
      } else {
        resolve(output)
      }
    })

    // Handle process errors
    pythonProcess.on("error", (err) => {
      reject(err)
    })
  })
}

