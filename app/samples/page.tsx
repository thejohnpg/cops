import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Card, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"

const sampleImages = [
  {
    id: "hubble-deep-field",
    title: "Hubble Deep Field",
    description: "One of the most famous images showing thousands of galaxies",
    path: "/samples/hubble-deep-field.jpg",
  },
  {
    id: "andromeda",
    title: "Andromeda Galaxy",
    description: "The nearest major galaxy to the Milky Way",
    path: "/samples/andromeda.jpg",
  },
  {
    id: "crab-nebula",
    title: "Crab Nebula",
    description: "A supernova remnant in the constellation of Taurus",
    path: "/samples/crab-nebula.jpg",
  },
  {
    id: "orion-nebula",
    title: "Orion Nebula",
    description: "A diffuse nebula situated in the Milky Way",
    path: "/samples/orion-nebula.jpg",
  },
  {
    id: "pillars-of-creation",
    title: "Pillars of Creation",
    description: "Elephant trunks of interstellar gas and dust in the Eagle Nebula",
    path: "/samples/pillars-of-creation.jpg",
  },
  {
    id: "whirlpool-galaxy",
    title: "Whirlpool Galaxy",
    description: "A grand-design spiral galaxy located in the constellation Canes Venatici",
    path: "/samples/whirlpool-galaxy.jpg",
  },
]

export default function SamplesPage() {
  return (
    <main className="container mx-auto py-8 px-4">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Sample Space Images</h1>
        <p className="text-gray-600">Use these sample images to test the celestial body detection system</p>
        <Button asChild className="mt-4">
          <Link href="/">Back to Home</Link>
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {sampleImages.map((image) => (
          <Card key={image.id} className="overflow-hidden">
            <div className="relative h-48 w-full">
              <Image src={image.path || "/placeholder.svg"} alt={image.title} fill className="object-cover" />
            </div>
            <CardHeader>
              <CardTitle>{image.title}</CardTitle>
              <CardDescription>{image.description}</CardDescription>
            </CardHeader>
            <CardFooter>
              <Button asChild variant="outline" className="w-full">
                <Link href={`/?image=${image.path}`}>Use This Image</Link>
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>
    </main>
  )
}

