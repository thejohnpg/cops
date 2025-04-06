import type React from "react"

export const metadata = {
  title: "Celestial Explorer - Space Image Processing",
  description:
    "Discover the universe through advanced image processing. Upload space images to detect and identify celestial bodies.",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>{/* Removendo o estilo inline para evitar problemas de hidratação */}</head>
      <body
        style={{
          margin: 0,
          padding: 0,
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
          backgroundColor: "#000",
          color: "#fff",
        }}
      >
        {children}
      </body>
    </html>
  )
}

