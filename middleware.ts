import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

// Este middleware está removendo qualquer redirecionamento para /inicio
export function middleware(request: NextRequest) {
  // Se a URL contiver /inicio, redirecione para a página inicial
  if (request.nextUrl.pathname.includes("/inicio")) {
    return NextResponse.redirect(new URL("/", request.url))
  }

  return NextResponse.next()
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
}

