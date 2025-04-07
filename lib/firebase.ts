import { initializeApp } from "firebase/app"
import { getStorage, ref, uploadBytes, getDownloadURL } from "firebase/storage"
import { getFirestore, collection, addDoc, serverTimestamp, doc, setDoc } from "firebase/firestore"

// Firebase configuration
const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_APIKEY,
  authDomain: process.env.NEXT_PUBLIC_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_MESSAGING_SENDER_ID,
  appId: process.env.NEXT_PUBLIC_APP_ID,
  measurementId: process.env.NEXT_PUBLIC_MEASUREMENT_ID,
}

// Initialize Firebase
const app = initializeApp(firebaseConfig)
const storage = getStorage(app)
const db = getFirestore(app)

// Upload image to Firebase Storage
export async function uploadImageToFirebase(file: File): Promise<string> {
  try {
    const timestamp = Date.now()
    const fileName = `celestial_images/${timestamp}_${file.name}`
    const storageRef = ref(storage, fileName)

    // Upload the file
    await uploadBytes(storageRef, file)

    // Get the download URL
    const downloadURL = await getDownloadURL(storageRef)
    return downloadURL
  } catch (error) {
    console.error("Error uploading image to Firebase:", error)
    throw error
  }
}

// Save analysis results to Firestore
export async function saveAnalysisToFirestore(results: any, imageUrl: string): Promise<string> {
  try {
    const analysisCollection = collection(db, "celestial_analyses")
    const docRef = await addDoc(analysisCollection, {
      ...results,
      imageUrl,
      timestamp: serverTimestamp(),
    })
    return docRef.id
  } catch (error) {
    console.error("Error saving analysis to Firestore:", error)
    throw error
  }
}

// Initialize COPS collection with required fields
export async function initializeCOPSCollection(): Promise<void> {
  try {
    // Create a document in the COPS collection
    const copsCollection = collection(db, "COPS")

    // Add a configuration document with default settings
    await setDoc(doc(copsCollection, "config"), {
      version: "1.0.0",
      lastUpdated: serverTimestamp(),
      settings: {
        defaultImageProcessingSettings: {
          enhanceContrast: true,
          detectStars: true,
          detectGalaxies: true,
          detectNebulae: true,
          annotateResults: true,
        },
        dataRetentionDays: 30,
        allowAnonymousUploads: true,
      },
    })

    console.log("COPS collection initialized successfully")
  } catch (error) {
    console.error("Error initializing COPS collection:", error)
    throw error
  }
}

// Call this function when the app initializes
try {
  initializeCOPSCollection()
} catch (error) {
  console.error("Failed to initialize COPS collection:", error)
}

export { storage, db }

