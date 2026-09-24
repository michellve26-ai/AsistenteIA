"""Preguntas frecuentes validadas para la interfaz de soporte.

Mantener este archivo separado permite actualizar las soluciones rápidas sin tocar
la lógica del chat/RAG.
"""

FAQS = [
    {
        "category": "Facturación",
        "question": "¿Cómo creo una factura de venta con crédito o vencimiento?",
        "keywords": "factura venta credito vencimiento plazo cliente facturar",
        "answer": """
Ve a **Crear > Factura de venta**.

1. Selecciona el cliente.
2. Define el punto de venta, entrega y lista de precios.
3. Agrega los productos o servicios.
4. Revisa impuestos, descuentos y demás datos.
5. Define el plazo/vencimiento correspondiente.
6. Pulsa **Facturar**.

La factura quedará con una cuenta por cobrar abierta y el vencimiento dependerá del plazo configurado.

> **Importante:** el nombre exacto de la opción para indicar un crédito específico puede variar según la configuración de la empresa.
""",
        "source": "BaseDeConociemiento.pdf — FAC-001, pág. 6",
    },
    {
        "category": "Facturación electrónica",
        "question": "¿Qué hago si se venció la resolución de facturación electrónica?",
        "keywords": "resolucion dian facturacion electronica rango prefijo consecutivo vencida",
        "answer": """
Primero debes tener autorizado el nuevo rango en la **DIAN** y luego asociarlo en OficinaPro.

1. Ingresa a **Configuraciones > Documentos electrónicos > Factura**.
2. Selecciona la oficina correspondiente y el nuevo prefijo/rango.
3. Indica el siguiente consecutivo real que vas a utilizar.
4. Pulsa **Asociar** para vincular la nueva resolución a la oficina.

> **Importante:** no inicies automáticamente en 1 si el rango ya tuvo uso. Una factura electrónica no debe emitirse con una resolución vencida ni con un consecutivo fuera del rango autorizado.
""",
        "source": "BaseDeConociemiento.pdf — FAC-007, pág. 12",
    },
    {
        "category": "Inventario",
        "question": "¿Cómo corrijo una cantidad equivocada en el inventario?",
        "keywords": "inventario corregir cantidad ajuste existencias producto kardex unidades",
        "answer": """
La corrección debe hacerse mediante un **ajuste de inventario**.

1. Ve a **Inventario > Ajustes inventario > Nuevo**.
2. Selecciona la oficina.
3. Elige el modo o tipo de ajuste.
4. Agrega el producto y coloca la cantidad o diferencia que necesitas corregir.
5. Guarda el ajuste.

> **Importante:** guardar el ajuste no necesariamente cambia la existencia de inmediato. Un usuario autorizado debe procesarlo/aprobarlo para que la existencia se actualice.
""",
        "source": "BaseDeConociemiento.pdf — INV-001, pág. 20",
    },
    {
        "category": "Cartera",
        "question": "¿Cómo consulto el estado de cuenta o cartera de un cliente?",
        "keywords": "cartera estado cuenta cliente saldo facturas vencimientos pagos retenciones",
        "answer": """
Puedes consultar la cuenta del cliente directamente:

1. Ve a **Usuarios > Clientes y Proveedores**.
2. Busca y selecciona el cliente.
3. Entra a su **cuenta/saldo**.

Allí puedes revisar facturas, vencimientos, valores pagados, retenciones y saldo pendiente. También puedes usar **Reportes > Cartera** para filtrar por cliente, fechas y oficina.

> El saldo puede incluir el efecto de pagos, notas y retenciones, por lo que no siempre coincide con el valor original de las facturas.
""",
        "source": "BaseDeConociemiento.pdf — CAR-001, pág. 19",
    },
    {
        "category": "Clientes y proveedores",
        "question": "¿Cómo edito un cliente o proveedor?",
        "keywords": "cliente proveedor tercero editar crear identificacion contacto usuarios",
        "answer": """
Para editar un tercero:

1. Ve a **Usuarios > Clientes y Proveedores**.
2. Busca y selecciona el cliente o proveedor.
3. Entra a su ficha.
4. Modifica los datos disponibles.
5. Guarda los cambios.

Si no aparece la opción de editar o no permite guardar, puede depender de los permisos del usuario, la oficina o la configuración de la empresa.
""",
        "source": "BaseDeConociemiento.pdf — CLI-001, pág. 18",
    },
    {
        "category": "Cajas y bancos",
        "question": "Al pagar solo me aparece efectivo, ¿cómo habilito un banco?",
        "keywords": "banco cuenta bancaria pagar pago efectivo caja oficina permisos",
        "answer": """
Si al registrar un pago solo aparece **efectivo**, primero debes crear o habilitar una cuenta bancaria para la oficina correspondiente.

1. Crea la cuenta asociada al banco desde la gestión de cajas/cuentas.
2. Verifica que la cuenta esté disponible para la oficina donde vas a registrar el pago.
3. Regresa al pago de la factura.
4. Revisa nuevamente el campo **Cuenta** y selecciona la cuenta bancaria.

Si la cuenta ya existe pero no aparece, revisa los **permisos del usuario** y la **oficina asignada a la cuenta**.
""",
        "source": "BaseDeConociemiento.pdf — CAJ-002, pág. 33",
    },
    {
        "category": "Notas crédito",
        "question": "¿Cómo reverso una nota crédito que ya fue aplicada o cerrada?",
        "keywords": "nota credito reversar aplicada cerrada anular factura ingresos",
        "answer": """
No es recomendable crear otra factura únicamente para “reversar” una nota crédito.

1. Ve a **Ingresos > Notas Crédito**.
2. Abre la nota crédito correspondiente.
3. Revisa si aparece la opción **Reversar**.
4. Si está disponible y tienes el permiso necesario, utiliza esa acción.

> Las acciones disponibles pueden cambiar según el estado del documento. No hagas ajustes manuales de inventario ni elimines la nota para compensarla.
""",
        "source": "BaseDeConociemiento.pdf — FAC-004, pág. 10",
    },
]
