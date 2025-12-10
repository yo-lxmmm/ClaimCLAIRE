"""Simple web application for baseline CLAIRE agent.

Environment Variables Required:
- GOOGLE_API_KEY: Your Google Gemini API key
- SERPER_API_KEY: Your Serper.dev API key (for web search)
"""

import asyncio
import json
import os
import queue
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Enable nested event loops for gRPC compatibility
import nest_asyncio
nest_asyncio.apply()

from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename

from claire_agent import InconsistencyAgent
from utils.logger import logger
from retrieval.retriever_client import set_search_before_date
from retrieval.source_trust_rating import get_source_trust_rating

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Uploads disabled - no backend storage configured
# if os.getenv('VERCEL_SERVERLESS') or os.getenv('VERCEL'):
#     app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
# else:
#     app.config['UPLOAD_FOLDER'] = 'uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['UPLOAD_FOLDER'] = '/tmp'  # Placeholder, uploads disabled

# Global agent instance
agent = None

# Thread pool for CPU-bound tasks and a dedicated asyncio loop for model calls
executor = ThreadPoolExecutor(max_workers=4)

def run_async_in_serverless(coro):
    """Run async coroutine in serverless environment using existing event loop."""
    # In serverless, we need to handle the case where there's already a running loop
    try:
        loop = asyncio.get_running_loop()
        # If we're already in an async context, we can't use run_until_complete
        # Create a new task and wait for it
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()
    except RuntimeError:
        # No running loop, safe to use asyncio.run
        return asyncio.run(coro)

# Create a single dedicated asyncio event loop in a background thread.
# All agent analyses will run on this loop to avoid cross-loop gRPC errors.
# Skip this in serverless environments (Vercel)
if not os.getenv('VERCEL_SERVERLESS'):
    analysis_loop = asyncio.new_event_loop()

    def _start_analysis_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    from threading import Thread
    analysis_thread = Thread(target=_start_analysis_loop, args=(analysis_loop,), daemon=True)
    analysis_thread.start()
else:
    # In serverless, use the current event loop
    analysis_loop = None

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests."""
    return '', 204

def initialize_agent():
    """Initialize the CLAIRE agent."""
    global agent
    try:
        agent = InconsistencyAgent(
            engine="gemini-2.5-flash",
            model_provider="google_genai",
            reasoning_effort=None,  # Google GenAI doesn't support reasoning effort
            num_results_per_query=10,  # Increased from 5 to get more sources, especially when date filtering
        )
        logger.info("Agent initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return False

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/single')
def single():
    """Single claim analysis page."""
    return render_template('single.html')

# Temporarily disabled
# @app.route('/batch')
# def batch():
#     """Batch analysis page."""
#     return render_template('batch.html')

@app.route('/status')
def status():
    """Check agent status."""
    return jsonify({
        "agent_initialized": agent is not None,
        "api_key_set": bool(os.getenv('GOOGLE_API_KEY')),
    })

@app.route('/analyze_single', methods=['POST'])
def analyze_single():
    """Analyze a single claim."""
    try:
        data = request.get_json()
        claim_text = data.get('claim', '').strip()
        before_date = (data.get('before_date') or '').strip() or None
        
        if not claim_text:
            return jsonify({'success': False, 'error': 'No claim provided'}), 400
        
        if not agent:
            return jsonify({'success': False, 'error': 'Agent not initialized'}), 500
        
        def run_analysis():
            """Submit analysis to the dedicated asyncio loop and wait for result."""
            async def analyze_async():
                try:
                    set_search_before_date(before_date)
                    return await agent.analyze_claim(claim_text, claim_text)
                finally:
                    set_search_before_date(None)

            try:
                if analysis_loop:
                    # Use background thread loop (local development)
                    fut = asyncio.run_coroutine_threadsafe(analyze_async(), analysis_loop)
                    return fut.result(timeout=120)
                else:
                    # Use run_async_in_serverless for serverless (Vercel)
                    return run_async_in_serverless(analyze_async())
            except Exception as e:
                logger.error(f"Error running analysis: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
        
        future = executor.submit(run_analysis)
        report = future.result(timeout=120)
        
        if report is None:
            return jsonify({'success': False, 'error': 'Analysis failed'}), 500
        
        # Extract search results with sources
        # Trust ratings are now computed during retrieval and stored in blocks
        # Parse the explanation to find which sources are referenced (e.g., [19, 21, 23])
        import re
        referenced_indices = set()
        if report.explanation:
            # Find all source references like [1], [19, 21, 23], [1, 2, 3]
            matches = re.findall(r'\[(\d+(?:,\s*\d+)*)\]', report.explanation)
            for match in matches:
                # Split by comma and extract numbers
                nums = [int(n.strip()) for n in match.split(',')]
                referenced_indices.update(nums)

        logger.info(f"Found {len(referenced_indices)} unique source references in explanation: {sorted(referenced_indices)}")

        sources = []
        if hasattr(report, 'search_results') and report.search_results:
            # Reorder sources: referenced ones first, then others
            # This ensures all sources mentioned in the explanation are visible
            all_blocks = report.search_results

            # Separate referenced and non-referenced blocks
            referenced_blocks = []
            other_blocks = []

            for idx, block in enumerate(all_blocks, 1):
                if idx in referenced_indices:
                    referenced_blocks.append((idx, block))
                else:
                    other_blocks.append((idx, block))

            # Sort referenced blocks by their original index
            referenced_blocks.sort(key=lambda x: x[0])

            # Combine: referenced first, then fill with others up to a limit
            max_sources_to_display = max(15, len(referenced_blocks))  # At least show all referenced + some extras
            blocks_to_process_with_indices = referenced_blocks + other_blocks
            blocks_to_process_with_indices = blocks_to_process_with_indices[:max_sources_to_display]

            logger.info(f"Displaying {len(blocks_to_process_with_indices)} sources ({len(referenced_blocks)} referenced, {len(blocks_to_process_with_indices) - len(referenced_blocks)} others)")

            # Extract just the blocks for processing
            blocks_to_process = [block for _, block in blocks_to_process_with_indices]
            original_indices = [idx for idx, _ in blocks_to_process_with_indices]
            
            # Check if blocks have trust ratings (they should from retrieval)
            # If not, compute them as fallback (for backward compatibility with cached results)
            blocks_needing_ratings = [b for b in blocks_to_process if not b.trust_rating]
            
            if blocks_needing_ratings:
                # Compute trust ratings only for blocks that don't have them
                logger.info(f"Computing trust ratings for {len(blocks_needing_ratings)} blocks missing ratings")
                async def compute_missing_trust_ratings():
                    """Compute trust ratings for blocks that don't have them."""
                    tasks = []
                    for block in blocks_needing_ratings:
                        task = get_source_trust_rating(
                            block.url or "",
                            block.document_title
                        )
                        tasks.append(task)
                    return await asyncio.gather(*tasks, return_exceptions=True)
                
                def get_trust_ratings():
                    if analysis_loop:
                        fut = asyncio.run_coroutine_threadsafe(compute_missing_trust_ratings(), analysis_loop)
                        return fut.result(timeout=30)
                    else:
                        return run_async_in_serverless(compute_missing_trust_ratings())
                
                try:
                    trust_ratings = executor.submit(get_trust_ratings).result(timeout=30)
                    # Update blocks with computed ratings
                    for block, trust_info in zip(blocks_needing_ratings, trust_ratings):
                        if isinstance(trust_info, Exception):
                            logger.warning(f"Error computing trust rating: {trust_info}")
                            block.trust_rating = "mixed"
                            block.trust_source = "llm"
                            block.trust_reason = "Error computing trust rating"
                            block.trust_confidence = 0.3  # Low confidence for errors
                        elif isinstance(trust_info, dict):
                            block.trust_rating = trust_info.get("rating", "mixed")
                            block.trust_source = trust_info.get("source", "list")
                            block.trust_reason = trust_info.get("reason", "")
                            block.trust_confidence = trust_info.get("confidence", 0.75)
                except Exception as e:
                    logger.warning(f"Error computing trust ratings: {e}, using defaults")
                    for block in blocks_needing_ratings:
                        block.trust_rating = "mixed"
                        block.trust_source = "llm"
                        block.trust_reason = "Error computing trust rating"
                        block.trust_confidence = 0.3  # Low confidence for errors
            
            # Build sources list from blocks (trust ratings already in blocks)
            # Use original indices and mark which ones are referenced
            for i, (block, original_idx) in enumerate(zip(blocks_to_process, original_indices), 1):
                # Extract domain name from URL
                domain_name = ""
                if block.url:
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(block.url)
                        domain = parsed.netloc.lower()
                        if domain.startswith('www.'):
                            domain = domain[4:]
                        domain_name = domain
                    except:
                        domain_name = ""

                # Check if this source was referenced in the explanation
                is_referenced = original_idx in referenced_indices

                sources.append({
                    "index": original_idx,  # Use original index from full list
                    "display_index": i,      # Position in displayed list
                    "title": block.document_title,
                    "domain": domain_name,
                    "url": block.url or "",
                    "section": block.section_title or "",
                    "preview": block.content[:200] + "..." if len(block.content) > 200 else block.content,
                    "trust_rating": block.trust_rating or "mixed",
                    "trust_source": block.trust_source or "list",
                    "trust_reason": block.trust_reason or "",
                    "trust_confidence": block.trust_confidence if hasattr(block, 'trust_confidence') and block.trust_confidence is not None else 0.75,
                    "referenced": is_referenced  # Flag if this source is mentioned in explanation
                })
        
        total_sources_found = len(report.search_results) if hasattr(report, 'search_results') and report.search_results else 0
        
        return jsonify({
            "success": True,
            "claim": claim_text,
            "verdict": report.verdict,
            "explanation": report.explanation,
            "sources": sources,
            "num_sources": len(sources),
            "total_sources_found": total_sources_found,  # Show total for transparency
        })
        
    except Exception as e:
        logger.error(f"Error analyzing claim: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analyze_single_stream', methods=['POST'])
def analyze_single_stream():
    """Analyze a single claim with real-time progress updates via Server-Sent Events."""
    try:
        data = request.get_json()
        claim_text = data.get('claim', '').strip()
        before_date = (data.get('before_date') or '').strip() or None

        if not claim_text:
            return jsonify({'success': False, 'error': 'No claim provided'}), 400

        if not agent:
            return jsonify({'success': False, 'error': 'Agent not initialized'}), 500

        # Create a queue for progress events
        event_queue = queue.Queue()

        def run_analysis_with_streaming():
            """Run analysis and emit progress events."""
            async def progress_callback(event):
                """Callback to capture progress events."""
                event_queue.put(event)

            async def analyze_async():
                try:
                    set_search_before_date(before_date)
                    return await agent.analyze_claim(
                        claim_text,
                        claim_text,
                        progress_callback=progress_callback
                    )
                finally:
                    set_search_before_date(None)

            try:
                if analysis_loop:
                    fut = asyncio.run_coroutine_threadsafe(analyze_async(), analysis_loop)
                    result = fut.result(timeout=120)
                else:
                    result = run_async_in_serverless(analyze_async())

                # Signal completion with final result including sources
                # Extract and format sources same as /analyze_single
                import re
                referenced_indices = set()
                if result.explanation:
                    matches = re.findall(r'\[(\d+(?:,\s*\d+)*)\]', result.explanation)
                    for match in matches:
                        nums = [int(n.strip()) for n in match.split(',')]
                        referenced_indices.update(nums)

                sources = []
                all_search_results = getattr(result, 'search_results', [])

                if all_search_results:
                    all_blocks = all_search_results
                    referenced_blocks = []
                    other_blocks = []

                    for idx, block in enumerate(all_blocks, 1):
                        if idx in referenced_indices:
                            referenced_blocks.append((idx, block))
                        else:
                            other_blocks.append((idx, block))

                    referenced_blocks.sort(key=lambda x: x[0])
                    max_sources_to_display = max(15, len(referenced_blocks))
                    blocks_to_process_with_indices = referenced_blocks + other_blocks
                    blocks_to_process_with_indices = blocks_to_process_with_indices[:max_sources_to_display]

                    for i, (original_idx, block) in enumerate(blocks_to_process_with_indices, 1):
                        domain_name = ""
                        if block.url:
                            try:
                                from urllib.parse import urlparse
                                parsed = urlparse(block.url)
                                domain = parsed.netloc.lower()
                                if domain.startswith('www.'):
                                    domain = domain[4:]
                                domain_name = domain
                            except:
                                domain_name = ""

                        is_referenced = original_idx in referenced_indices

                        sources.append({
                            "index": original_idx,
                            "display_index": i,
                            "title": block.document_title,
                            "domain": domain_name,
                            "url": block.url or "",
                            "section": block.section_title or "",
                            "preview": block.content[:200] + "..." if len(block.content) > 200 else block.content,
                            "trust_rating": block.trust_rating or "mixed",
                            "trust_source": block.trust_source or "list",
                            "trust_reason": block.trust_reason or "",
                            "trust_confidence": block.trust_confidence if hasattr(block, 'trust_confidence') and block.trust_confidence is not None else 0.75,
                            "referenced": is_referenced
                        })

                event_queue.put({
                    'type': 'complete',
                    'data': {
                        'verdict': result.verdict,
                        'explanation': result.explanation,
                        'wording_feedback': result.wording_feedback,
                        'sources': sources,
                        'num_sources': len(sources),
                        'total_sources_found': len(all_search_results),
                    }
                })
                return result
            except Exception as e:
                logger.error(f"Error in streaming analysis: {e}")
                import traceback
                logger.error(traceback.format_exc())
                event_queue.put({'type': 'error', 'message': str(e)})
                return None

        def generate():
            """Generator for Server-Sent Events."""
            # Start analysis in background thread
            executor.submit(run_analysis_with_streaming)

            while True:
                try:
                    # Wait for events with timeout
                    event = event_queue.get(timeout=0.5)

                    # Send event to frontend
                    yield f"data: {json.dumps(event)}\n\n"

                    # Break if complete or error
                    if event.get('type') in ['complete', 'error']:
                        break

                except queue.Empty:
                    # Send keepalive
                    yield f": keepalive\n\n"
                    continue

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive',
            }
        )

    except Exception as e:
        logger.error(f"Error in stream setup: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Temporarily disabled
# @app.route('/upload_analyze', methods=['POST'])
# def upload_analyze():
#     """Upload file and analyze claims. ENABLED for local use."""
#     try:
#         if 'claims_file' not in request.files:
#             return jsonify({'success': False, 'error': 'No file uploaded'}), 400
#
#         file = request.files['claims_file']
#         if file.filename == '':
#             return jsonify({'success': False, 'error': 'No file selected'}), 400
#
#         if not file.filename.endswith('.json'):
#             return jsonify({'success': False, 'error': 'Only JSON files are supported'}), 400
#
#         # Read file directly without saving (better for local use)
#         claims_data = json.load(file.stream)
#
#         if not isinstance(claims_data, list):
#             return jsonify({'success': False, 'error': 'JSON must be a list of claims'}), 400
#
#         # Analyze claims
#         def run_analyses():
#             """Run batch analyses on the dedicated asyncio loop."""
#             results = []
#             for item in claims_data:
#                 claim_text = item.get('claim', '')
#                 passage = item.get('passage', claim_text)
#                 before_date_item = (item.get('before_date') or item.get('before') or '').strip() if isinstance(item, dict) else ''
#                 before_date_item = before_date_item or None
#
#                 async def analyze_single_claim():
#                     try:
#                         set_search_before_date(before_date_item)
#                         reports = await agent.analyze_passage_for_inconsistencies(passage=claim_text)
#                         if reports and len(reports) > 0:
#                             return reports[0]
#                         return None
#                     finally:
#                         set_search_before_date(None)
#
#                 try:
#                     if analysis_loop:
#                         fut = asyncio.run_coroutine_threadsafe(analyze_single_claim(), analysis_loop)
#                         report = fut.result(timeout=180)
#                     else:
#                         report = asyncio.run(analyze_single_claim())
#
#                     if report is None:
#                         results.append({"claim": claim_text, "verdict": "error", "error": "Analysis failed"})
#                     else:
#                         results.append({
#                             "claim": claim_text,
#                             "verdict": report.verdict,
#                             "explanation": report.explanation[:200] + "..." if len(report.explanation) > 200 else report.explanation,
#                             "search_results_count": len(report.search_results) if hasattr(report, 'search_results') else 0,
#                         })
#                 except Exception as e:
#                     logger.error(f"Error running analysis for claim '{claim_text[:50]}...': {e}")
#                     results.append({"claim": claim_text, "verdict": "error", "error": str(e)})
#
#             return results
#
#         future = executor.submit(run_analyses)
#         results = future.result(timeout=600)  # 10 minute timeout
#
#         return jsonify({
#             "success": True,
#             "results": results,
#             "total": len(results),
#         })
#
#     except json.JSONDecodeError:
#         return jsonify({'success': False, 'error': 'Invalid JSON file'}), 400
#     except Exception as e:
#         logger.error(f"Error in upload analysis: {e}")
#         return jsonify({'success': False, 'error': str(e)}), 500

# Temporarily disabled
# @app.route('/download_template')
# def download_template():
#     """Download a template JSON file."""
#     template = [
#         {
#             "claim": "The first human to walk on the moon was Neil Armstrong in 1969",
#             "passage": "Context about the moon landing and Neil Armstrong",
#             "before_date": ""
#         },
#         {
#             "claim": "Taylor Swift is engaged",
#             "passage": "Taylor Swift engagement rumors",
#             "before_date": "2025-08-26"
#         },
#         {
#             "claim": "Python is a compiled programming language",
#             "passage": "Context about Python programming language",
#             "before_date": ""
#         }
#     ]
#
#     # Use /tmp for template file (works in serverless)
#     template_file = '/tmp/claims_template.json'
#     with open(template_file, 'w') as f:
#         json.dump(template, f, indent=2)
#
#     return send_file(template_file, as_attachment=True, download_name='claims_template.json')

@app.route('/timeline')
def timeline():
    """Timeline analysis page."""
    return render_template('timeline.html')

@app.route('/analyze_timeline', methods=['POST'])
def analyze_timeline():
    """
    Analyze claim verdict changes over time using binary search.

    Request body:
    {
        "claim": "The claim to analyze",
        "start_year": 2019,
        "end_year": 2024,
        "precision": "month"  // optional: "month", "week", "day"
    }
    """
    try:
        if agent is None:
            return jsonify({'success': False, 'error': 'Agent not initialized'}), 503

        data = request.get_json()
        claim = data.get('claim', '').strip()
        start_date = data.get('start_date', '').strip()
        end_date = data.get('end_date', '').strip()
        precision = data.get('precision', 'month')

        # Validate inputs
        if not claim:
            return jsonify({'success': False, 'error': 'Claim is required'}), 400

        if not start_date or not end_date:
            return jsonify({'success': False, 'error': 'Start date and end date are required'}), 400

        # Validate date range
        from datetime import datetime
        try:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

        if start_dt >= end_dt:
            return jsonify({'success': False, 'error': 'Start date must be before end date'}), 400

        # Check time range (max 50 years)
        days_diff = (end_dt - start_dt).days
        if days_diff > 50 * 365:
            return jsonify({'success': False, 'error': 'Time range too large (max 50 years)'}), 400

        if precision not in ['month', 'week', 'day']:
            precision = 'month'

        # Import enhanced timeline analyzer
        from claire_agent.timeline.enhanced_analyzer import build_enhanced_timeline

        logger.info(f"Enhanced timeline analysis request: {claim} ({start_date} to {end_date})")

        # Run enhanced timeline analysis on dedicated loop
        async def run_timeline_analysis():
            """Run enhanced timeline analysis asynchronously."""
            import time
            start_time = time.time()

            result = await build_enhanced_timeline(
                claim=claim,
                start_date=start_date,
                end_date=end_date,
                agent=agent
            )

            execution_time = time.time() - start_time
            result.metadata['execution_time'] = execution_time

            return result

        # Execute on the dedicated loop
        if analysis_loop:
            fut = asyncio.run_coroutine_threadsafe(run_timeline_analysis(), analysis_loop)
            result = fut.result(timeout=600)  # 10 minute timeout
        else:
            result = run_async_in_serverless(run_timeline_analysis())

        logger.info(f"Timeline analysis complete: {result.metadata.get('events_verified', 0)} events verified")

        return jsonify({
            'success': True,
            **result.to_dict()
        })

    except asyncio.TimeoutError:
        logger.error("Timeline analysis timed out")
        return jsonify({'success': False, 'error': 'Analysis timed out (>10 minutes). Try a smaller time range.'}), 504
    except Exception as e:
        logger.error(f"Timeline analysis error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    if initialize_agent():
        print("🚀 Starting Baseline CLAIRE Web App...")
        print("📱 Open your browser and go to: http://localhost:8080")
        print("📤 You can upload JSON files with claims to analyze")
        # Disable reloader to avoid event loop conflicts with gRPC
        app.run(debug=True, host='0.0.0.0', port=8080, use_reloader=False)
    else:
        print("❌ Failed to initialize agent. Please check your API key.")

