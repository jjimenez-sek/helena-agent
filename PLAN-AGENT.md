# Ver solo archivos modificados
git show --name-only 1943a8f518fb98005d5f0e304d3c2486a27fde2c
commit 1943a8f518fb98005d5f0e304d3c2486a27fde2c (HEAD -> func/rfcRecords, origin/func/rfcRecords)
Author: kh31r0n <jorge.jimenez107@gmail.com>
Date:   Wed May 13 13:45:41 2026 -0500

    Implement records for the backend

src/agents/rfc_reuse_confirm.py
src/agents/rfc_reuse_validate.py
src/agents/triage.py
src/graphs/main_graph.py
src/graphs/state.py
src/main.py
kheiron@r3s3n:~/helena/helena-agent-proof$ git show 1943a8f518fb98005d5f0e304d3c2486a27fde2c
commit 1943a8f518fb98005d5f0e304d3c2486a27fde2c (HEAD -> func/rfcRecords, origin/func/rfcRecords)
Author: kh31r0n <jorge.jimenez107@gmail.com>
Date:   Wed May 13 13:45:41 2026 -0500

    Implement records for the backend

diff --git a/src/agents/rfc_reuse_confirm.py b/src/agents/rfc_reuse_confirm.py
new file mode 100644
index 0000000..3e61178
--- /dev/null
+++ b/src/agents/rfc_reuse_confirm.py
@@ -0,0 +1,172 @@
+import structlog
+from langchain_core.messages import AIMessage
+from langchain_core.runnables import RunnableConfig
+from langgraph.config import get_stream_writer
+
+from ..graphs.state import AgentState
+from ..llm import get_openai_client, resolve_api_key
+from ..observability import get_langfuse, record_node_invocation
+
+logger = structlog.get_logger(__name__)
+
+_CONFIRM_KEYWORDS = {"confirm", "yes", "approve", "submit", "ok", "okay", "sí", "si", "confirmar", "enviar"}
+
+
+async def rfc_reuse_confirm_node(
+    state: AgentState,
+    config: RunnableConfig,
+) -> dict:
+    record_node_invocation("rfc_reuse_confirm")
+
+    api_key: str = resolve_api_key(config)
+    client = get_openai_client(api_key)
+    thread_id: str = state.get("thread_id", "unknown")
+    rfc_data: dict = dict(state.get("rfc_template_data") or {})
+    conversation_id: str = state.get("conversation_id", "")
+    project_workflows: list = list(state.get("project_workflows") or [])
+
+    langfuse = get_langfuse()
+    trace = langfuse.trace(
+        name="rfc_reuse_confirm",
+        metadata={"thread_id": thread_id, "node": "rfc_reuse_confirm"},
+    )
+
+    write = get_stream_writer()
+    write({"type": "rfc_step_progress", "step": 2, "total_open_steps": 2, "topic": "Confirmation"})
+
+    has_new_message = (
+        bool(state["messages"])
+        and getattr(state["messages"][-1], "type", "") == "human"
+    )
+
+    rfc_reuse_confirmed = False
+    rfc_reuse_validated = state.get("rfc_reuse_validated", True)
+    is_correction = False
+
+    if has_new_message:
+        user_text = (state["messages"][-1].content or "").strip().lower()
+        if any(kw in user_text for kw in _CONFIRM_KEYWORDS):
+            rfc_reuse_confirmed = True
+            logger.info("rfc_reuse_confirmed_by_user", thread_id=thread_id)
+        else:
+            # User sent a correction — invalidate so we go back to validate
+            is_correction = True
+            rfc_reuse_validated = False
+            logger.info("rfc_reuse_correction_received", thread_id=thread_id)
+
+    if rfc_reuse_confirmed:
+        # Emit execute_workflow and acknowledge
+        workflows_to_run = [
+            {"workflow_id": wf["workflow_id"], "name": wf.get("name", "workflow")}
+            for wf in project_workflows
+            if wf.get("workflow_id")
+        ]
+
+        write({
+            "type": "execute_workflow",
+            "conversation_id": conversation_id,
+            "workflows": workflows_to_run,
+            "rfc_data": rfc_data,
+        })
+
+        logger.info(
+            "rfc_reuse_execute_triggered",
+            thread_id=thread_id,
+            conversation_id=conversation_id,
+            workflow_count=len(workflows_to_run),
+        )
+
+        # Stream brief confirmation
+        confirm_messages = [
+            {
+                "role": "system",
+                "content": (
+                    "The user has confirmed the RFC from a previous template. "
+                    "Acknowledge briefly that the RFC has been confirmed and workflows are being triggered. "
+                    "Be concise. Respond in the same language the user writes in (likely Spanish)."
+                ),
+            },
+        ]
+
+        gen = trace.generation(
+            name="rfc_reuse_confirm_llm",
+            model="gpt-5",
+            input={"messages": confirm_messages},
+        )
+
+        stream = await client.chat.completions.create(
+            model="gpt-5",
+            messages=confirm_messages,
+            stream=True,
+            stream_options={"include_usage": True},
+        )
+
+        full_response = ""
+        prompt_tokens = 0
+        completion_tokens = 0
+
+        async for chunk in stream:
+            delta = chunk.choices[0].delta.content if chunk.choices else ""
+            if delta:
+                write({"type": "token", "content": delta})
+                full_response += delta
+            if chunk.usage:
+                prompt_tokens = chunk.usage.prompt_tokens
+                completion_tokens = chunk.usage.completion_tokens
+
+        gen.end(output=full_response, usage={"input": prompt_tokens, "output": completion_tokens})
+
+        return {
+            "messages": [AIMessage(content=full_response)],
+            "rfc_reuse_confirmed": True,
+            "rfc_execute_confirmed": True,
+        }
+
+    if is_correction:
+        # Acknowledge correction — routing will send back to validate
+        correction_messages = [
+            {
+                "role": "system",
+                "content": (
+                    "The user wants to make changes to their RFC before confirming. "
+                    "Acknowledge that you received their corrections and that you will re-validate the RFC. "
+                    "Be concise. Respond in the same language the user writes in (likely Spanish)."
+                ),
+            },
+        ]
+
+        gen = trace.generation(
+            name="rfc_reuse_correction_ack_llm",
+            model="gpt-5",
+            input={"messages": correction_messages},
+        )
+
+        stream = await client.chat.completions.create(
+            model="gpt-5",
+            messages=correction_messages,
+            stream=True,
+            stream_options={"include_usage": True},
+        )
+
+        full_response = ""
+        prompt_tokens = 0
+        completion_tokens = 0
+
+        async for chunk in stream:
+            delta = chunk.choices[0].delta.content if chunk.choices else ""
+            if delta:
+                write({"type": "token", "content": delta})
+                full_response += delta
+            if chunk.usage:
+                prompt_tokens = chunk.usage.prompt_tokens
+                completion_tokens = chunk.usage.completion_tokens
+
+        gen.end(output=full_response, usage={"input": prompt_tokens, "output": completion_tokens})
+
+        return {
+            "messages": [AIMessage(content=full_response)],
+            "rfc_reuse_validated": False,  # Force re-validation
+        }
+
+    # No user message yet — should not normally happen, but handle gracefully
+    return {"messages": []}
diff --git a/src/agents/rfc_reuse_validate.py b/src/agents/rfc_reuse_validate.py
new file mode 100644
index 0000000..09db933
--- /dev/null
+++ b/src/agents/rfc_reuse_validate.py
@@ -0,0 +1,251 @@
+import json
+
+import structlog
+from langchain_core.messages import AIMessage
+from langchain_core.runnables import RunnableConfig
+from langgraph.config import get_stream_writer
+
+from ..graphs.state import AgentState
+from ..llm import get_openai_client, resolve_api_key
+from ..observability import get_langfuse, record_node_invocation
+
+logger = structlog.get_logger(__name__)
+
+# All fields that must be present for a complete RFC
+REQUIRED_OPEN_FIELDS = [
+    "rfc_number", "client_name", "client_authorizer",
+    "technical_responsible_name", "change_supervisor",
+    "work_team", "change_objective", "technical_description",
+    "infrastructure_involved", "impact_during_change",
+    "impact_after_change", "systems_involved",
+    "implementation_steps", "estimated_duration",
+    "rollback_conditions", "rollback_procedure",
+    "technical_validations", "acceptance_criteria",
+]
+
+REQUIRED_CLOSED_FIELDS = [
+    "change_type", "service_impact", "monitoring_loss",
+    "remote_access_loss", "backup_status", "rollback_available",
+    "risk_level", "execution_location",
+]
+
+_VALIDATION_SYSTEM_PROMPT = """You are Helena, a security operations AI assistant.
+
+You are reviewing a pre-filled RFC (Request for Change) that a user wants to resubmit based on a previous RFC.
+
+Your job:
+1. Check if the data is COMPLETE — all required fields must have meaningful, non-placeholder values.
+2. Check for LOGICAL CONSISTENCY:
+   - If risk_level is "Alto" or "Crítico", there MUST be a detailed rollback_procedure and rollback_conditions.
+   - If service_impact is "Impacto total", implementation_steps should mention a maintenance window.
+   - If backup_status is "No", rollback_procedure must be especially detailed.
+   - If change_type is "Cambio de emergencia", estimated_duration should be reasonable for urgency.
+3. Check for STALE or PLACEHOLDER text — generic phrases like "TBD", "TODO", "same as before", or copy-paste artifacts.
+
+Return ONLY a JSON object (no markdown, no extra text):
+{
+  "valid": true/false,
+  "missing_fields": ["field_name", ...],
+  "inconsistencies": ["description of issue", ...],
+  "suggestions": ["suggestion for improvement", ...]
+}
+
+If valid is true, missing_fields and inconsistencies must be empty arrays.
+"""
+
+_PRESENT_VALID_PROMPT = """You are Helena, a security operations AI assistant.
+
+The user has submitted a pre-filled RFC based on a previous one. The data has been validated and is COMPLETE and CONSISTENT.
+
+Present a clean, professional summary of the RFC data below, organized by sections.
+After the summary, ask: "Does this RFC look correct? Reply **confirm** to submit it, or tell me what needs to be corrected."
+
+Respond in the same language the user writes in (likely Spanish).
+"""
+
+_PRESENT_INVALID_PROMPT = """You are Helena, a security operations AI assistant.
+
+The user submitted a pre-filled RFC but it has issues that need to be resolved before submission.
+
+Validation results:
+{validation_json}
+
+Current RFC data:
+{rfc_data_str}
+
+Present the issues clearly:
+1. List any missing fields and ask the user to provide them.
+2. List any logical inconsistencies and explain what needs to change.
+3. List any suggestions for improvement.
+
+Be helpful and specific. Tell the user exactly what information you need.
+Respond in the same language the user writes in (likely Spanish).
+"""
+
+
+async def rfc_reuse_validate_node(
+    state: AgentState,
+    config: RunnableConfig,
+) -> dict:
+    record_node_invocation("rfc_reuse_validate")
+
+    api_key: str = resolve_api_key(config)
+    client = get_openai_client(api_key)
+    thread_id: str = state.get("thread_id", "unknown")
+    rfc_data: dict = dict(state.get("rfc_template_data") or {})
+
+    langfuse = get_langfuse()
+    trace = langfuse.trace(
+        name="rfc_reuse_validate",
+        metadata={"thread_id": thread_id, "node": "rfc_reuse_validate"},
+    )
+
+    write = get_stream_writer()
+    write({"type": "rfc_step_progress", "step": 1, "total_open_steps": 2, "topic": "Validation"})
+
+    # If user sent a new message (correction after failed validation), merge it
+    has_new_message = (
+        bool(state["messages"])
+        and getattr(state["messages"][-1], "type", "") == "human"
+    )
+
+    if has_new_message and not state.get("rfc_reuse_validated", False):
+        # User is providing missing info — use LLM to extract and merge
+        user_text = state["messages"][-1].content or ""
+        extraction_messages = [
+            {
+                "role": "system",
+                "content": (
+                    "The user is providing missing or corrected information for an RFC. "
+                    "Extract any field values from their message and return a JSON object "
+                    "mapping field names (snake_case) to their values. "
+                    "Only include fields the user explicitly provided. "
+                    "Return ONLY JSON, no markdown."
+                ),
+            },
+            {
+                "role": "user",
+                "content": f"Current RFC data:\n{json.dumps(rfc_data, indent=2, ensure_ascii=False)}\n\nUser correction:\n{user_text}",
+            },
+        ]
+
+        extraction_resp = await client.chat.completions.create(
+            model="gpt-5",
+            messages=extraction_messages,
+        )
+        try:
+            extracted = json.loads(extraction_resp.choices[0].message.content.strip())
+            if isinstance(extracted, dict):
+                rfc_data.update(extracted)
+                logger.info("rfc_reuse_correction_merged", thread_id=thread_id, fields=list(extracted.keys()))
+        except (json.JSONDecodeError, AttributeError):
+            logger.warning("rfc_reuse_correction_parse_failed", thread_id=thread_id)
+
+    # Step 1: Field presence check
+    missing_fields = []
+    for field in REQUIRED_OPEN_FIELDS + REQUIRED_CLOSED_FIELDS:
+        value = rfc_data.get(field)
+        if not value or (isinstance(value, str) and not value.strip()):
+            missing_fields.append(field)
+
+    # Step 2: LLM consistency check
+    rfc_data_str = json.dumps(rfc_data, indent=2, ensure_ascii=False)
+
+    validation_messages = [
+        {"role": "system", "content": _VALIDATION_SYSTEM_PROMPT},
+        {"role": "user", "content": f"RFC data to validate:\n\n{rfc_data_str}"},
+    ]
+
+    gen = trace.generation(
+        name="rfc_reuse_validate_llm",
+        model="gpt-5",
+        input={"messages": validation_messages},
+    )
+
+    validation_resp = await client.chat.completions.create(
+        model="gpt-5",
+        messages=validation_messages,
+    )
+
+    validation_text = validation_resp.choices[0].message.content or "{}"
+    gen.end(output=validation_text, usage={
+        "input": validation_resp.usage.prompt_tokens if validation_resp.usage else 0,
+        "output": validation_resp.usage.completion_tokens if validation_resp.usage else 0,
+    })
+
+    try:
+        validation_result = json.loads(validation_text.strip())
+    except (json.JSONDecodeError, AttributeError):
+        logger.warning("rfc_reuse_validation_parse_failed", thread_id=thread_id, raw=validation_text[:200])
+        validation_result = {
+            "valid": False,
+            "missing_fields": missing_fields,
+            "inconsistencies": ["Could not parse validation response"],
+            "suggestions": [],
+        }
+
+    # Merge programmatic missing fields with LLM-detected ones
+    llm_missing = validation_result.get("missing_fields", [])
+    all_missing = list(set(missing_fields + llm_missing))
+    is_valid = validation_result.get("valid", False) and len(all_missing) == 0
+
+    logger.info(
+        "rfc_reuse_validation_result",
+        thread_id=thread_id,
+        valid=is_valid,
+        missing_count=len(all_missing),
+        inconsistency_count=len(validation_result.get("inconsistencies", [])),
+    )
+
+    # Step 3: Stream the response
+    if is_valid:
+        rfc_data_str_formatted = "\n".join(f"- **{k}**: {v}" for k, v in rfc_data.items())
+        response_messages = [
+            {"role": "system", "content": _PRESENT_VALID_PROMPT},
+            {"role": "user", "content": f"RFC data:\n\n{rfc_data_str_formatted}"},
+        ]
+    else:
+        validation_result["missing_fields"] = all_missing
+        response_messages = [
+            {
+                "role": "system",
+                "content": _PRESENT_INVALID_PROMPT.format(
+                    validation_json=json.dumps(validation_result, indent=2, ensure_ascii=False),
+                    rfc_data_str="\n".join(f"- **{k}**: {v}" for k, v in rfc_data.items()),
+                ),
+            },
+        ]
+
+    gen2 = trace.generation(
+        name="rfc_reuse_validate_response_llm",
+        model="gpt-5",
+        input={"messages": response_messages},
+    )
+
+    stream = await client.chat.completions.create(
+        model="gpt-5",
+        messages=response_messages,
+        stream=True,
+        stream_options={"include_usage": True},
+    )
+
+    full_response = ""
+    prompt_tokens = 0
+    completion_tokens = 0
+
+    async for chunk in stream:
+        delta = chunk.choices[0].delta.content if chunk.choices else ""
+        if delta:
+            write({"type": "token", "content": delta})
+            full_response += delta
+        if chunk.usage:
+            prompt_tokens = chunk.usage.prompt_tokens
+            completion_tokens = chunk.usage.completion_tokens
+
+    gen2.end(output=full_response, usage={"input": prompt_tokens, "output": completion_tokens})
+
+    return {
+        "messages": [AIMessage(content=full_response)],
+        "rfc_template_data": rfc_data,
+        "rfc_reuse_validated": is_valid,
+    }
diff --git a/src/agents/triage.py b/src/agents/triage.py
index d61b3ca..fa559ca 100644
--- a/src/agents/triage.py
+++ b/src/agents/triage.py
@@ -47,6 +47,10 @@ async def triage_node(
     ):
         return {}
 
+    # If in RFC reuse flow, skip re-classification
+    if state.get("intent") == "rfc_reuse" and state.get("rfc_reuse_mode", False):
+        return {}
+
     api_key: str = resolve_api_key(config)
     client = get_openai_client(api_key)
     thread_id: str = state.get("thread_id", "unknown")
@@ -103,7 +107,7 @@ async def triage_node(
         logger.warning("triage_json_parse_failed", raw=full_response[:200])
         intent = "unknown"
 
-    valid_intents = {"rfc", "incident", "knowledge", "escalation", "unknown"}
+    valid_intents = {"rfc", "rfc_reuse", "incident", "knowledge", "escalation", "unknown"}
     if intent not in valid_intents:
         intent = "unknown"
 
@@ -115,7 +119,11 @@ async def triage_node(
 
 def route_from_triage(
     state: AgentState,
-) -> Literal["rfc_open_questions", "rfc_closed_questions", "rfc_summary_confirm", "rfc_execute", "fallback_response"]:
+) -> Literal[
+    "rfc_open_questions", "rfc_closed_questions", "rfc_summary_confirm", "rfc_execute",
+    "rfc_reuse_validate", "rfc_reuse_confirm",
+    "fallback_response",
+]:
     intent = state.get("intent", "unknown")
 
     if intent == "rfc":
@@ -135,4 +143,13 @@ def route_from_triage(
             return "rfc_closed_questions"
         return "rfc_open_questions"
 
+    if intent == "rfc_reuse":
+        if state.get("rfc_execute_confirmed", False):
+            return "fallback_response"
+        if not state.get("rfc_reuse_validated", False):
+            return "rfc_reuse_validate"
+        if not state.get("rfc_reuse_confirmed", False):
+            return "rfc_reuse_confirm"
+        return "fallback_response"
+
     return "fallback_response"
diff --git a/src/graphs/main_graph.py b/src/graphs/main_graph.py
index c630dc4..7732970 100644
--- a/src/graphs/main_graph.py
+++ b/src/graphs/main_graph.py
@@ -6,11 +6,15 @@ from ..agents.fallback import fallback_response_node
 from ..agents.rfc_closed import rfc_closed_questions_node
 from ..agents.rfc_execute import rfc_execute_node
 from ..agents.rfc_open import rfc_open_questions_node
+from ..agents.rfc_reuse_confirm import rfc_reuse_confirm_node
+from ..agents.rfc_reuse_validate import rfc_reuse_validate_node
 from ..agents.rfc_summary import rfc_summary_confirm_node
 from ..agents.triage import route_from_triage, triage_node
 from .state import AgentState
 
 
+# ── Original RFC flow routing ────────────────────────────────────────────────
+
 def _route_from_rfc_open(state: AgentState) -> Literal["rfc_closed_questions", "__end__"]:
     """After open questions: if all done, proceed to closed questions immediately."""
     if state.get("rfc_open_complete"):
@@ -26,15 +30,40 @@ def _route_from_rfc_closed(state: AgentState) -> Literal["rfc_summary_confirm",
 
 
 def _route_from_rfc_summary(state: AgentState) -> Literal["rfc_execute", "__end__"]:
-    """After summary confirmation: proceed to execute phase (which will present workflows and ask for final confirm)."""
+    """After summary confirmation: proceed to execute phase."""
     if state.get("rfc_confirmed"):
         return "rfc_execute"
     return END
 
 
+# ── RFC reuse flow routing ───────────────────────────────────────────────────
+
+def _route_from_rfc_reuse_validate(
+    state: AgentState,
+) -> Literal["rfc_reuse_confirm", "__end__"]:
+    """After validation: if valid, proceed to confirmation. Otherwise wait for user input."""
+    if state.get("rfc_reuse_validated"):
+        return "rfc_reuse_confirm"
+    return END
+
+
+def _route_from_rfc_reuse_confirm(
+    state: AgentState,
+) -> Literal["rfc_reuse_validate", "__end__"]:
+    """After confirmation: if user corrected, re-validate. Otherwise done (execute already emitted or waiting)."""
+    if state.get("rfc_reuse_confirmed"):
+        return END  # execute_workflow already emitted
+    if not state.get("rfc_reuse_validated"):
+        return "rfc_reuse_validate"  # Re-validate after correction
+    return END  # Wait for user confirmation
+
+
+# ── Graph construction ───────────────────────────────────────────────────────
+
 def build_graph(checkpointer):
     graph = StateGraph(AgentState)
 
+    # Original nodes
     graph.add_node("triage", triage_node)
     graph.add_node("rfc_open_questions", rfc_open_questions_node)
     graph.add_node("rfc_closed_questions", rfc_closed_questions_node)
@@ -42,6 +71,10 @@ def build_graph(checkpointer):
     graph.add_node("rfc_execute", rfc_execute_node)
     graph.add_node("fallback_response", fallback_response_node)
 
+    # RFC reuse nodes
+    graph.add_node("rfc_reuse_validate", rfc_reuse_validate_node)
+    graph.add_node("rfc_reuse_confirm", rfc_reuse_confirm_node)
+
     graph.add_edge(START, "triage")
 
     graph.add_conditional_edges(
@@ -52,6 +85,8 @@ def build_graph(checkpointer):
             "rfc_closed_questions": "rfc_closed_questions",
             "rfc_summary_confirm": "rfc_summary_confirm",
             "rfc_execute": "rfc_execute",
+            "rfc_reuse_validate": "rfc_reuse_validate",
+            "rfc_reuse_confirm": "rfc_reuse_confirm",
             "fallback_response": "fallback_response",
         },
     )
@@ -67,7 +102,6 @@ def build_graph(checkpointer):
         _route_from_rfc_closed,
         {"rfc_summary_confirm": "rfc_summary_confirm", END: END},
     )
-
     graph.add_conditional_edges(
         "rfc_summary_confirm",
         _route_from_rfc_summary,
@@ -76,4 +110,16 @@ def build_graph(checkpointer):
     graph.add_edge("rfc_execute", END)
     graph.add_edge("fallback_response", END)
 
+    # RFC reuse edges
+    graph.add_conditional_edges(
+        "rfc_reuse_validate",
+        _route_from_rfc_reuse_validate,
+        {"rfc_reuse_confirm": "rfc_reuse_confirm", END: END},
+    )
+    graph.add_conditional_edges(
+        "rfc_reuse_confirm",
+        _route_from_rfc_reuse_confirm,
+        {"rfc_reuse_validate": "rfc_reuse_validate", END: END},
+    )
+
     return graph.compile(checkpointer=checkpointer)
diff --git a/src/graphs/state.py b/src/graphs/state.py
index 09e4d2a..5eb7939 100644
--- a/src/graphs/state.py
+++ b/src/graphs/state.py
@@ -20,3 +20,8 @@ class AgentState(TypedDict):
     rfc_closed_complete: bool
     rfc_confirmed: bool
     rfc_execute_confirmed: bool  # True once user confirms the execute step (prevents re-execution)
+    # RFC reuse (template) flow
+    rfc_template_data: dict      # Pre-filled data from a previous RFC
+    rfc_reuse_mode: bool         # True when using template flow
+    rfc_reuse_validated: bool    # LLM validation passed
+    rfc_reuse_confirmed: bool    # User confirmed after validation
diff --git a/src/main.py b/src/main.py
index 1188dbf..b36d4ac 100644
--- a/src/main.py
+++ b/src/main.py
@@ -97,6 +97,16 @@ class ChatStreamRequest(BaseModel):
     project_steps: list = []  # [{type, order, content?, workflows?}]
 
 
+class ChatFromTemplateRequest(BaseModel):
+    thread_id: str
+    message: str
+    openai_api_key: str
+    project_id: str = ""
+    conversation_id: str = ""
+    project_steps: list = []
+    rfc_template_data: dict = {}
+
+
 class ChatResumeRequest(BaseModel):
     thread_id: str
     interrupt_id: str
@@ -325,6 +335,73 @@ async def chat_stream(
     )
 
 
+@app.post("/chat/stream-from-template")
+async def chat_stream_from_template(
+    req: ChatFromTemplateRequest,
+    user_sub: Annotated[str, Depends(get_current_user)],
+) -> StreamingResponse:
+    """Start an RFC reuse flow with pre-filled template data.
+
+    Reuses the same _stream_graph() generator as /chat/stream, inheriting
+    all timeout, client-disconnect detection, SSE formatting, and error handling.
+    Only the initial state differs: intent is pre-set to "rfc_reuse".
+    """
+    if compiled_graph is None:
+        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent graph not initialised")
+
+    thread_id = scoped_thread_id(user_sub, req.thread_id)
+    agent_requests_total.inc()
+
+    config: dict = {
+        "configurable": {
+            "thread_id": thread_id,
+            "openai_api_key": req.openai_api_key,
+        }
+    }
+
+    project_workflows = [
+        {
+            "workflow_id": wf.get("workflowId"),
+            "name": wf.get("workflowName"),
+            "description": wf.get("description"),
+        }
+        for step in req.project_steps
+        if step.get("type") == "WORKFLOW"
+        for wf in step.get("workflows", [])
+        if wf.get("workflowId")
+    ]
+
+    inputs: dict = {
+        "messages": [HumanMessage(content=req.message)],
+        "thread_id": thread_id,
+        "project_id": req.project_id,
+        "conversation_id": req.conversation_id,
+        "project_workflows": project_workflows,
+        # RFC reuse-specific fields
+        "intent": "rfc_reuse",
+        "rfc_reuse_mode": True,
+        "rfc_template_data": req.rfc_template_data,
+    }
+
+    logger.info(
+        "chat_stream_from_template_start",
+        thread_id=thread_id,
+        user_sub=user_sub,
+        project_id=req.project_id,
+        workflow_count=len(project_workflows),
+        template_field_count=len(req.rfc_template_data),
+    )
+
+    return StreamingResponse(
+        _stream_graph(inputs, config),
+        media_type="text/event-stream",
+        headers={
+            "Cache-Control": "no-cache",
+            "X-Accel-Buffering": "no",
+        },
+    )
+
+
 @app.post("/chat/resume")
 async def chat_resume(
     req: ChatResumeRequest,

