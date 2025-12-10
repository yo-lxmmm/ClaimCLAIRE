"""Source trust rating module.

Determines trust ratings for sources based on known domain lists and LLM classification.
"""

import asyncio
import re
from typing import Optional, Literal
from urllib.parse import urlparse
import os
import logging
import json

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from json_repair import repair_json
import diskcache
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Initialize cache for LLM classifications
# In serverless (Vercel), use /tmp which is writable
if os.getenv('VERCEL_SERVERLESS') or os.getenv('VERCEL'):
    _cache_dir = "/tmp/source_trust_ratings"
else:
    _cache_dir = os.path.join(os.path.dirname(__file__), "..", ".cache", "source_trust_ratings")
    # Create cache directory if it doesn't exist (local development)
    os.makedirs(_cache_dir, exist_ok=True)

_llm_cache = diskcache.Cache(_cache_dir)

# Reusable LLM model instance (initialized lazily)
_llm_model = None

# Trust rating weights for explicit numeric weighting
TRUST_WEIGHTS = {
    "reliable": 1.0,
    "mixed": 0.5,
    "unreliable": 0.1
}
VALID_TRUST_RATINGS = set(TRUST_WEIGHTS.keys())

# Default confidence scores by source type
DEFAULT_CONFIDENCE = {
    "list": 0.95,  # High confidence for curated lists
    "llm": 0.75,   # Moderate confidence for LLM classification
    "error": 0.3   # Low confidence for errors
}


class TrustRatingResponse(BaseModel):
    """Structured response from LLM for trust rating classification."""
    rating: Literal["reliable", "mixed", "unreliable"] = Field(
        description="Trust rating category"
    )
    reason: str = Field(
        description="Brief 1-2 sentence explanation for the rating"
    )
    confidence: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0"
    )


def _get_llm_model():
    """Get or create the LLM model instance (singleton pattern)."""
    global _llm_model
    if _llm_model is None:
        _llm_model = init_chat_model(
            model="gemini-2.5-flash",
            model_provider="google_genai",
            temperature=0,  # Deterministic output
        )
    return _llm_model

# Trust rating categories based on domains
GENERALLY_RELIABLE = {
    "abcnews.com",
    "afp.com",
    "aljazeera.com",
    "amnesty.org",
    "aon.com",
    "ap.org",
    "arstechnica.com",
    "atlasobscura.com/articles",
    "avclub.com",
    "avn.com",
    "axios.com",
    "bbc.co.uk",
    "behindthevoiceactors.com",
    "bellingcat.com",
    "burkespeerage.com",
    "buzzfeednews.com",
    "cbsnews.com",
    "channelnewsasia.com",
    "checkyourfact.com",
    "climatefeedback.org",
    "cnn.com",
    "codastory.com",
    "commonsensemedia.org",
    "csmonitor.com",
    "deadline.com",
    "debretts.com",
    "denofgeek.com",
    "deseretnews.com",
    "digitalspy.co.uk",
    "digitaltrends.com",
    "dw.com/en",
    "economist.com",
    "engadget.com",
    "eurogamer.net",
    "ew.com",
    "ft.com",
    "gamedeveloper.com",
    "gameinformer.com",
    "gamespot.com",
    "gizmodo.com",
    "glaad.org",
    "gq.com",
    "haaretz.com",
    "hardcoregaming101.net",
    "hollywoodreporter.com",
    "idolator.com",
    "ifcncodeofprinciples.poynter.org",
    "ign.com",
    "independent.co.uk",
    "indianexpress.com",
    "ipsnews.net",
    "iranicaonline.org",
    "jamanetwork.com",
    "jpost.com",
    "kirkusreviews.com",
    "kommersant.ru",
    "latimes.com",
    "lwn.net",
    "meduza.io",
    "metacritic.com",
    "mg.co.za",
    "monde-diplomatique.fr",
    "motherjones.com",
    "msnbc.com",
    "nationalgeographic.com",
    "nationalpost.com",
    "nbcnews.com",
    "newrepublic.com",
    "news.sky.com",
    "news.yahoo.com",
    "newslaundry.com",
    "newsnationnow.com",
    "newyorker.com",
    "nme.com",
    "npr.org",
    "nydailynews.com",
    "nymag.com",
    "nytimes.com",
    "nzherald.co.nz",
    "oko.press",
    "pbs.org",
    "people.com",
    "pewresearch.org",
    "pinknews.co.uk",
    "playboy.com",
    "politico.com",
    "politifact.com",
    "propublica.org",
    "rappler.com",
    "reason.com",
    "reuters.com",
    "rfa.org",
    "rollingstone.com",
    "rottentomatoes.com",
    "rte.ie/",
    "scientificamerican.com",
    "scmp.com",
    "scotusblog.com",
    "skepticalinquirer.org",
    "smh.com.au",
    "snopes.com",
    "space.com",
    "spiegel.de",
    "splcenter.org",
    "straitstimes.com",
    "theage.com.au",
    "theatlantic.com",
    "theaustralian.com.au",
    "theconversation.com",
    "thediplomat.com",
    "theglobeandmail.com",
    "thehill.com",
    "thehindu.com",
    "theinsneider.com",
    "theintercept.com",
    "themarysue.com",
    "thenation.com",
    "theregister.co.uk",
    "thetimes.com",
    "theverge.com",
    "thewire.in",
    "thewrap.com",
    "time.com",
    "timesofisrael.com",
    "torrentfreak.com",
    "tvguide.com",
    "usatoday.com",
    "usnews.com",
    "vanityfair.com",
    "variety.com",
    "villagevoice.com",
    "voanews.com",
    "vogue.com",
    "vox.com",
    "washingtonpost.com",
    "weeklystandard.com",
    "wired.com",
    "wsj.com",
    "wyborcza.pl",
}

MIXED_CONSENSUS = {
    "about.com",
    "academia.edu",
    "alarabiya.net",
    "alexa.com",
    "allmusic.com",
    "allsides.com",
    "aninews.in",
    "arabnews.com",
    "askmen.com",
    "aspi.org.au",
    "ballotpedia.org",
    "boingboing.net",
    "britannica.com",
    "bustle.com",
    "buzzfeed.com",
    "cato.org",
    "cbn.com",
    "cepr.net",
    "chinadaily.com.cn",
    "coindesk.com",
    "cosmopolitan.com",
    "dailydot.com",
    "dailynk.com/english/",
    "democracynow.org",
    "destructoid.com",
    "dexerto.com",
    "encyclopedia.com",
    "entrepreneur.com",
    "euromedmonitor.org",
    "fair.org",
    "genius.com",
    "guinnessworldrecords.com",
    "heavy.com",
    "hk.appledaily.com",
    "hopenothate.org.uk",
    "humanevents.com",
    "ijr.com",
    "islamqa.info",
    "jacobinmag.com",
    "jezebel.com",
    "lapatilla.com",
    "maps.google.com",
    "mdpi.com",
    "mediaite.com",
    "mediamatters.org",
    "memri.org",
    "metalsucks.net",
    "middleeastmonitor.com",
    "mirror.co.uk",
    "mondoweiss.net",
    "morningstaronline.co.uk",
    "nationalreview.com",
    "parliament.uk",
    "polygon.com",
    "popularmechanics.com",
    "pride.com",
    "quackwatch.org",
    "realclearpolitics.com",
    "rferl.org",
    "rian.ru",
    "salon.com",
    "sciencebasedmedicine.org",
    "scienceblogs.com",
    "screenrant.com",
    "sherdog.com",
    "si.com",
    "skepdic.com",
    "skynews.com.au",
    "socialblade.com",
    "sparknotes.com",
    "spectator.co.uk",
    "standard.co.uk",
    "techcrunch.com",
    "ted.com",
    "telegraph.co.uk",
    "theamericanconservative.com",
    "thearda.com",
    "thedailybeast.com",
    "thefp.com",
    "thegreenpapers.com",
    "theguardian.com",
    "thejc.com",
    "theneedledrop.com",
    "thenextweb.com",
    "thesouthafrican.com",
    "thinkprogress.org",
    "timesofindia.com",
    "tmz.com",
    "townhall.com",
    "trtworld.com",
    "usmagazine.com",
    "venturebeat.com",
    "vice.com",
    "washingtonexaminer.com",
    "washingtontimes.com",
    "worldchristiandatabase.org",
    "wsws.org",
    "www.astronautix.com",
    "xbiz.com",
    "xinhuanet.com",
}

GENERALLY_UNRELIABLE = {
    "112.ua",
    "Ladbible.com",
    "TV.com",
    "aa.com.tr/en",
    "acclaimedmusic.net",
    "adfontesmedia.com",
    "adl.org",
    "allmovie.com",
    "almanar.com.lb",
    "alternet.org",
    "amazon.com",
    "ancestry.com",
    "anphoblacht.com",
    "answers.com",
    "antiwar.com",
    "armyrecognition.com",
    "arxiv.org",
    "atlasobscura.com/places",
    "benzinga.com",
    "bild.de",
    "blogspot.com",
    "broadwayworld.com",
    "californiaglobe.com",
    "catholic-hierarchy.org",
    "celebritynetworth.com",
    "cesnur.org",
    "cnet.com",
    "consortiumnews.com",
    "correodelorinoco.gob.ve",
    "council.rollingstone.com",
    "counterpunch.org",
    "cracked.com",
    "dailykos.com",
    "dailysabah.com",
    "dailywire.com",
    "deviantart.com",
    "discogs.com",
    "distractify.com",
    "dorchesterreview.ca",
    "electronicintifada.net",
    "ethnicelebs.com",
    "eurasiantimes.com",
    "express.co.uk",
    "facebook.com",
    "familysearch.org",
    "fandom.com",
    "faroutmagazine.co.uk",
    "filmaffinity.com",
    "findagrave.com",
    "findmypast.co.uk",
    "flickr.com",
    "forbes.com",
    "forbes.com/advisor",
    "fotw.info",
    "foxnews.com",
    "freebeacon.com",
    "gawker.com",
    "gbnews.uk",
    "geonames.nga.mil",
    "geonames.usgs.gov",
    "globalsecurity.org",
    "goodreads.com",
    "heatst.com",
    "history.com",
    "huffpost.com",
    "ibtimes.com",
    "imdb.com",
    "inaturalist.org",
    "indymedia.org",
    "inquisitr.com",
    "instagram.com",
    "investopedia.com",
    "jewishvirtuallibrary.org",
    "joshuaproject.net",
    "jrank.org",
    "knowyourmeme.com",
    "kotaku.com",
    "landtransportguru.net",
    "linkedin.com",
    "lionheartv.net",
    "livejournal.com",
    "marquiswhoswho.com",
    "mashable.com",
    "meaww.com",
    "mediabiasfactcheck.com",
    "medium.com",
    "metal-experience.com",
    "metro.co.uk",
    "mrc.org",
    "newsnationnow.com/space/ufo*",
    "ngo-monitor.org",
    "nypost.com",
    "order-order.com",
    "ourcampaigns.com",
    "panampost.com",
    "patheos.com",
    "pinkvilla.com",
    "planespotters.net",
    "prnewswire.com",
    "quadrant.org.au",
    "quillette.com",
    "quora.com",
    "rawstory.com",
    "reddit.com",
    "redstate.com",
    "rollingstone.com/politics",
    "sciencedirect.com/topics/",
    "scribd.com",
    "sixthtone.com",
    "skwawkbox.org",
    "sourcewatch.org",
    "spirit-of-metal.com",
    "sportskeeda.com",
    "stackexchange.com",
    "starsunfolded.com",
    "statista.com",
    "tass.com",
    "theblaze.com",
    "thecanary.co",
    "thefederalist.com",
    "thenewamerican.com",
    "theonion.com",
    "thepostmillennial.com",
    "thetruthaboutguns.com",
    "townandvillageguide.com",
    "tvtropes.org",
    "twitter.com",
    "ukwhoswho.com",
    "urbandictionary.com",
    "venezuelanalysis.com",
    "vgchartz.com",
    "victimsofcommunism.org",
    "watchmojo.com",
    "weather2travel.com",
    "wegotthiscovered.com",
    "westernjournal.com",
    "whatculture.com",
    "whosampled.com",
    "wikidata.org",
    "wikileaks.org",
    "wikinews.org",
    "wikipedia.org",
    "wordpress.com",
    "worldometers.info",
    "youtube.com",
    "zdnet.com",
    "almayadeen.net",
    "anna-news.info",
    "baike.baidu.com",
    "cgtn.com",
    "crunchbase.com",
    "dailycaller.com",
    "dailymail.co.uk",
    "dailystar.co.uk",
    "eadaily.com",
    "frontpagemag.com",
    "genealogy.eu",
    "geni.com",
    "globaltimes.cn",
    "hispantv.com",
    "jihadwatch.org",
    "journal-neo.org",
    "journalofscientificexploration.org",
    "last.fm",
    "lifesitenews.com",
    "mailonsunday.co.uk",
    "martinoticias.com",
    "metal-archives.com",
    "mintpressnews.com",
    "nationalenquirer.com",
    "newsblaze.com",
    "newsbreak.com",
    "newsmax.com",
    "newsoftheworld.co.uk",
    "nndb.com",
    "oann.com",
    "occupydemocrats.com",
    "presstv.com",
    "rateyourmusic.com",
    "republicworld.com",
    "royalcentral.co.uk",
    "rt.com",
    "simpleflying.com",
    "sputniknews.com",
    "takimag.com",
    "tasnimnews.com",
    "telesurtv.net",
    "thecradle.co",
    "thedebrief.org",
    "theepochtimes.com",
    "thegatewaypundit.com",
    "thegrayzone.com",
    "thesun.co.uk",
    "unz.com",
    "vdare.com",
    "voltairenet.org",
    "wenweipo.com",
    "wnd.com",
    "zerohedge.com",
    "bestgore.com",
    "breitbart.com",
    "change.org",
    "company-histories.com",
    "examiner.com",
    "famousbirthdays.com",
    "globalresearch.ca",
    "gofundme.com",
    "healthline.com",
    "heritage.org",
    "indiegogo.com",
    "infowars.com",
    "kickstarter.com",
    "lenta.ru",
    "liveleak.com",
    "lulu.com",
    "mylife.com",
    "naturalnews.com",
    "news-front.su",
    "opindia.com",
    "projectveritas.com",
    "southfront.org",
    "swarajyamag.com",
    "thepointsguy.com",
    "thepointsguy.com/news",
    "veteranstoday.com",
    "zoominfo.com",
}


def extract_domain_from_url(url: str) -> Optional[str]:
    """Extract the domain from a URL.
    
    Args:
        url: The URL to extract domain from
        
    Returns:
        The domain (e.g., "bbc.co.uk") or None if parsing fails
    """
    try:
        if not url:
            return None
        
        # Ensure URL has a scheme
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return domain if domain else None
    except Exception as e:
        logger.debug(f"Error extracting domain from URL '{url}': {e}")
        return None


def match_domain_pattern(domain: str, pattern: str) -> bool:
    """Check if a domain matches a pattern.
    
    Patterns can be exact domains or include wildcards (*).
    Examples:
        - "bbc.co.uk" matches "bbc.co.uk" or "www.bbc.co.uk"
        - "dw.com/en" matches "dw.com" with path /en
        - "newsnationnow.com/space/ufo*" matches UFO paths
    
    Args:
        domain: The full domain + path to check
        pattern: The pattern to match against
        
    Returns:
        True if domain matches pattern
    """
    # Handle exact matches first
    if domain == pattern:
        return True
    
    # Handle patterns with paths (e.g., "dw.com/en")
    if '/' in pattern:
        # Pattern includes a path, check if domain starts with it
        return domain.startswith(pattern)
    
    # Handle wildcard patterns (e.g., "newsnationnow.com/space/ufo*")
    if '*' in pattern:
        pattern_regex = pattern.replace('*', '.*')
        return bool(re.match(f"^{pattern_regex}", domain))
    
    # For patterns without paths, match the domain part (ignoring subdomains)
    # e.g., "en.wikipedia.org" should match "wikipedia.org"
    domain_only = domain.split('/')[0]  # Get domain without path
    
    # Exact match
    if domain_only == pattern:
        return True
    
    # Check if domain ends with pattern (to handle subdomains like "en.wikipedia.org")
    # But ensure it's a proper subdomain by requiring a dot before the pattern
    # e.g., "en.wikipedia.org" ends with ".wikipedia.org" ✓
    # but "royalsociety.org" does NOT end with ".society.org" ✗
    if domain_only.endswith('.' + pattern):
        return True
    
    return False


def get_trust_rating_from_lists(url: str, title: str = "") -> Optional[dict]:
    """Check if source is in the known trust lists.
    
    Args:
        url: The URL of the source
        title: Optional title (not used in domain-based matching)
        
    Returns:
        dict with "rating" and "confidence" if found, None otherwise
    """
    domain = extract_domain_from_url(url)
    if not domain:
        return None
    
    # Get full domain + path for pattern matching
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        parsed = urlparse(url)
        domain_with_path = parsed.netloc.lower()
        if domain_with_path.startswith('www.'):
            domain_with_path = domain_with_path[4:]
        if parsed.path and parsed.path != '/':
            domain_with_path += parsed.path.rstrip('/')
    except:
        domain_with_path = domain
    
    # Check against each list in order of precedence:
    # 1. Unreliable (highest precedence - most specific patterns)
    # 2. Reliable (second precedence)
    # 3. Mixed (lowest precedence - catch-all for ambiguous sources)
    # This ensures domains in multiple lists get the most appropriate rating
    
    for pattern in GENERALLY_UNRELIABLE:
        if match_domain_pattern(domain_with_path, pattern.lower()):
            return {
                "rating": "unreliable",
                "confidence": DEFAULT_CONFIDENCE["list"]
            }
    
    for pattern in GENERALLY_RELIABLE:
        if match_domain_pattern(domain_with_path, pattern.lower()):
            return {
                "rating": "reliable",
                "confidence": DEFAULT_CONFIDENCE["list"]
            }
    
    for pattern in MIXED_CONSENSUS:
        if match_domain_pattern(domain_with_path, pattern.lower()):
            return {
                "rating": "mixed",
                "confidence": DEFAULT_CONFIDENCE["list"]
            }
    
    return None


async def classify_source_with_llm(url: str, title: str = "") -> dict:
    """Use LLM to classify a source into one of three trust categories.

    Args:
        url: The URL of the source
        title: Optional title of the article

    Returns:
        dict with:
            - "rating": "reliable", "mixed", or "unreliable"
            - "reason": explanation
    """
    domain = extract_domain_from_url(url)

    # Check cache first
    cache_key = f"llm_classification:{domain}"
    cached_result = _llm_cache.get(cache_key)
    if cached_result is not None:
        logger.info(f"Using cached LLM classification for {domain}: {cached_result['rating']}")
        return cached_result

    # Construct the classification prompt with explicit numeric weights
    prompt = f"""You are a fact-checking expert evaluating source credibility.

Classify the following source into ONE of these categories:
- "reliable" (weight: {TRUST_WEIGHTS['reliable']}): Generally trustworthy news sources with strong editorial standards (e.g., major newspapers, established news agencies, peer-reviewed journals)
- "mixed" (weight: {TRUST_WEIGHTS['mixed']}): Sources with mixed reliability or potential bias (e.g., opinion-heavy sites, some partisan media, aggregators)
- "unreliable" (weight: {TRUST_WEIGHTS['unreliable']}): Generally unreliable sources (e.g., tabloids, social media, user-generated content sites, known misinformation sources)

Source to classify:
Domain: {domain}
URL: {url}
{f'Title: {title}' if title else ''}

Consider:
1. Editorial standards and fact-checking processes
2. Track record of accuracy
3. Journalistic ethics and transparency
4. Potential biases or conflicts of interest

Respond ONLY with a JSON object in this exact format:
{{
    "rating": "reliable" | "mixed" | "unreliable",
    "reason": "Brief 1-2 sentence explanation",
    "confidence": 0.0-1.0
}}"""

    try:
        # Get reusable model instance with structured output
        model = _get_llm_model()
        structured_model = model.with_structured_output(TrustRatingResponse)

        # Make the LLM call with timeout
        logger.info(f"Requesting LLM classification for {domain}")
        
        response: TrustRatingResponse = await asyncio.wait_for(
            structured_model.ainvoke([HumanMessage(content=prompt)]),
            timeout=10.0  # 10 second timeout
        )

        # Validate confidence is in range
        confidence = max(0.0, min(1.0, response.confidence))

        logger.info(f"LLM classified {domain} as '{response.rating}' (confidence: {confidence:.2f}): {response.reason}")

        result_dict = {
            "rating": response.rating,
            "reason": response.reason,
            "confidence": confidence
        }

        # Cache the result for 7 days
        _llm_cache.set(cache_key, result_dict, expire=7 * 24 * 60 * 60)

        return result_dict

    except asyncio.TimeoutError:
        logger.warning(f"LLM classification timeout for {url}")
        return {
            "rating": "mixed",
            "confidence": DEFAULT_CONFIDENCE["error"],
            "reason": "LLM classification timed out - defaulting to mixed reliability"
        }
    except json.JSONDecodeError as e:
        logger.warning(f"LLM JSON parse error for {url}: {e}")
        return {
            "rating": "mixed",
            "confidence": DEFAULT_CONFIDENCE["error"],
            "reason": f"Failed to parse LLM response - defaulting to mixed reliability"
        }
    except Exception as e:
        # If anything goes wrong, default to "mixed" reliability with low confidence
        logger.warning(f"LLM classification failed for {url}: {e}")
        return {
            "rating": "mixed",
            "confidence": DEFAULT_CONFIDENCE["error"],
            "reason": f"Unable to classify - defaulting to mixed reliability (error: {str(e)[:100]})"
        }


async def get_source_trust_rating(url: str, title: str = "") -> dict:
    """Get trust rating for a source.
    
    First checks against known domain lists, then uses LLM if not found.
    
    Args:
        url: The URL of the source
        title: Optional title of the article
        
    Returns:
        dict with keys:
            - "rating": "reliable", "mixed", or "unreliable"
            - "source": "list" or "llm"
            - "confidence": float between 0.0 and 1.0
            - "reason": explanation (only if from LLM)
    """
    # First, check against known lists
    list_result = get_trust_rating_from_lists(url, title)
    if list_result:
        return {
            "rating": list_result["rating"],
            "source": "list",
            "confidence": list_result["confidence"]
        }
    
    # If not found, use LLM to classify
    try:
        llm_result = await classify_source_with_llm(url, title)
    except Exception as exc:
        logger.warning(f"LLM trust rating failure for {url}: {exc}")
        return {
            "rating": "mixed",
            "source": "llm",
            "confidence": 0.4,
            "reason": f"LLM classification error - defaulting to mixed ({str(exc)[:80]})"
        }

    rating = llm_result.get("rating")
    if rating not in VALID_TRUST_RATINGS:
        logger.warning(f"Invalid trust rating '{rating}' for {url}; defaulting to mixed.")
        rating = "mixed"

    raw_confidence = llm_result.get("confidence", DEFAULT_CONFIDENCE["llm"])
    confidence = min(raw_confidence if raw_confidence is not None else DEFAULT_CONFIDENCE["llm"], 0.75)

    if confidence < 0.3:
        return {
            "rating": "mixed",
            "source": "llm",
            "confidence": 0.5,
            "reason": "LLM confidence too low (<0.3); defaulting to mixed reliability"
        }

    return {
        "rating": rating,
        "source": "llm",
        "confidence": confidence,
        "reason": llm_result.get("reason", "")
    }
