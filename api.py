# =========================================================
# MANUALES EXTRA (subida manual de PDF)
# =========================================================


def _exigir_clave(token: str | None) -> None:
    """Si UPLOAD_TOKEN está definido, exige la clave."""

    if not token_valido(token):
        raise HTTPException(
            status_code=401,
            detail="Se requiere la clave para modificar los manuales extra.",
        )


@app.get("/api/extra-docs")
def listar_extra_docs():
    """Manuales extra subidos manualmente."""

    return {
        "documentos": extra_docs.listar(),
        "max_mb": MAX_UPLOAD_MB,
    }


@app.post("/api/extra-docs")
async def subir_extra_doc(
    request: Request,
    nombre: str,
    x_upload_token: str | None = Header(default=None),
):
    """
    Recibe un PDF (cuerpo de la petición = bytes del archivo) y lo agrega
    como referencia extra, sin tocar la indexación automática.
    """

    _exigir_clave(x_upload_token)

    limite = (
        f"El PDF supera el máximo permitido ({MAX_UPLOAD_MB} MB)."
    )

    declarado = request.headers.get("content-length", "")

    if declarado.isdigit() and int(declarado) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=limite)

    recibido = bytearray()

    async for bloque in request.stream():
        recibido.extend(bloque)

        if len(recibido) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=limite)

    if not recibido:
        raise HTTPException(
            status_code=400,
            detail="No se recibió ningún archivo.",
        )

    try:
        resultado = await run_in_threadpool(
            extra_docs.agregar_pdf,
            nombre,
            bytes(recibido),
        )

    except ExtraDocsError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception:
        logger.exception("Error indexando un manual extra.")

        raise HTTPException(
            status_code=500,
            detail="No fue posible indexar el documento.",
        )

    return {"ok": True, **resultado}


@app.delete("/api/extra-docs/{nombre}")
def eliminar_extra_doc(
    nombre: str,
    x_upload_token: str | None = Header(default=None),
):
    """Quita un manual extra (no afecta los PDF de data/pdfs)."""

    _exigir_clave(x_upload_token)

    quitados = extra_docs.eliminar(nombre)

    if quitados is None:
        raise HTTPException(
            status_code=404,
            detail="No existe ese manual extra.",
        )

    return {"ok": True, "fragmentos_quitados": quitados}


# =========================================================
# INTERFAZ ESTÁTICA
# =========================================================

# IMPORTANTE:
#
# StaticFiles debe mantenerse AL FINAL.
#
# Si se monta antes de /api/* podría interceptar
# las rutas del backend.

app.mount(
    "/",
    StaticFiles(
        directory=BASE_DIR / "static",
        html=True,
    ),
    name="static",
)
