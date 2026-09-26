import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowRight,
  Check,
  ChevronDown,
  ChevronRight,
  Heart,
  Search,
  ShoppingBag,
  SlidersHorizontal,
  Sparkles,
  Star,
  X,
  Zap,
} from "lucide-react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

const CATEGORY_CONFIG = [
  { label: "Laptops", value: "laptops", icon: "⌁" },
  { label: "Mobiles", value: "mobiles", icon: "▣" },
  { label: "Electronics", value: "electronics", icon: "◉" },
  { label: "Fashion", value: "fashion", icon: "◇" },
  { label: "Shoes", value: "shoes", icon: "◌" },
  { label: "Accessories", value: "accessories", icon: "○" },
];

const SUGGESTIONS = [
  "Lightweight laptop for software development under ₹50,000",
  "Phone with great camera under ₹20,000",
  "Blue formal shirt for office",
];

const formatPrice = (price) => `₹${Number(price || 0).toLocaleString("en-IN")}`;

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [intent, setIntent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [wishlist, setWishlist] = useState([]);
  const [cart, setCart] = useState([]);

  const [selectedProduct, setSelectedProduct] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [recommendationLoading, setRecommendationLoading] = useState(false);
  const [recommendationError, setRecommendationError] = useState("");

  const [activePanel, setActivePanel] = useState(null);
  const [showFilters, setShowFilters] = useState(false);
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [priceFilter, setPriceFilter] = useState("all");
  const [sortBy, setSortBy] = useState("relevance");
  const [browseMode, setBrowseMode] = useState(false);
  const [toast, setToast] = useState("");

  const [displayText, setDisplayText] = useState("");
  const fullText = "Find products that fit your life.";

  useEffect(() => {
    let index = 0;
    const typing = window.setInterval(() => {
      index += 1;
      setDisplayText(fullText.slice(0, index));
      if (index >= fullText.length) window.clearInterval(typing);
    }, 42);

    return () => window.clearInterval(typing);
  }, []);

  useEffect(() => {
    if (!toast) return undefined;
    const timer = window.setTimeout(() => setToast(""), 2400);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    document.body.style.overflow = selectedProduct || activePanel ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [selectedProduct, activePanel]);

  const getProductImage = (category) => {
    const normalized = category?.toLowerCase().trim();
    const imageMap = {
      laptops: "/products/laptops/laptop-default.png",
      mobiles: "/products/mobiles/mobile-default.png",
      electronics: "/products/electronics/electronics-default.png",
      fashion: "/products/fashion/fashion-default.png",
      shoes: "/products/shoes/shoes-default.png",
      accessories: "/products/accessories/accessories-default.png",
    };
    return imageMap[normalized] || null;
  };

  const registerProduct = (product) => product;

  const toggleWishlist = (product) => {
    if (!product?.product_id) return;

    setWishlist((current) => {
      const exists = current.some((item) => item.product_id === product.product_id);
      if (exists) {
        setToast("Removed from wishlist");
        return current.filter((item) => item.product_id !== product.product_id);
      }
      setToast("Added to wishlist");
      return [...current, registerProduct(product)];
    });
  };

  const addToBag = (product) => {
    if (!product?.product_id) return;

    setCart((current) => {
      const existing = current.find((item) => item.product_id === product.product_id);
      if (existing) {
        return current.map((item) =>
          item.product_id === product.product_id
            ? { ...item, quantity: item.quantity + 1 }
            : item,
        );
      }
      return [...current, { ...product, quantity: 1 }];
    });

    setToast(`${product.product_name} added to bag`);
  };

  const removeFromBag = (productId) => {
    setCart((current) => current.filter((item) => item.product_id !== productId));
    setToast("Removed from bag");
  };

  const changeQuantity = (productId, delta) => {
    setCart((current) =>
      current
        .map((item) =>
          item.product_id === productId
            ? { ...item, quantity: Math.max(0, item.quantity + delta) }
            : item,
        )
        .filter((item) => item.quantity > 0),
    );
  };

  const scrollTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const handleSearch = async (searchQuery = query) => {
    const trimmedQuery = searchQuery.trim();
    if (!trimmedQuery || loading) return;

    setQuery(trimmedQuery);
    setBrowseMode(false);
    setLoading(true);
    setError("");
    setActivePanel(null);

    try {
      const response = await fetch(`${API_BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: trimmedQuery, top_k: 5 }),
      });

      if (!response.ok) throw new Error("Search service unavailable");

      const data = await response.json();
      setResults(data.results || []);
      setIntent(data.intent || null);

      window.setTimeout(() => scrollTo("results"), 80);
    } catch (err) {
      console.error("Search error:", err);
      setResults([]);
      setIntent(null);
      setError("We couldn't complete your search right now. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const openProduct = async (product) => {
    if (!product) return;

    setSelectedProduct(product);
    setRecommendations(null);
    setRecommendationError("");
    setRecommendationLoading(true);

    try {
      const response = await fetch(`${API_BASE}/recommendations/${product.product_id}`);
      if (!response.ok) throw new Error("Recommendation service unavailable");

      const data = await response.json();
      setRecommendations(data);
    } catch (err) {
      console.error("Recommendation error:", err);
      setRecommendationError("Recommendations are currently unavailable.");
    } finally {
      setRecommendationLoading(false);
    }
  };

  const closeProduct = () => {
    setSelectedProduct(null);
    setRecommendations(null);
    setRecommendationError("");
  };

  useEffect(() => {
    const handleEscape = (event) => {
      if (event.key !== "Escape") return;
      if (selectedProduct) closeProduct();
      else setActivePanel(null);
    };

    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [selectedProduct]);

  const visibleResults = useMemo(() => {
    let filtered = [...results];

    if (categoryFilter !== "all") {
      filtered = filtered.filter(
        (product) => product.category?.toLowerCase() === categoryFilter,
      );
    }

    if (priceFilter !== "all") {
      const max = Number(priceFilter);
      filtered = filtered.filter((product) => Number(product.price) <= max);
    }

    if (sortBy === "price-low") {
      filtered.sort((a, b) => Number(a.price) - Number(b.price));
    } else if (sortBy === "price-high") {
      filtered.sort((a, b) => Number(b.price) - Number(a.price));
    } else if (sortBy === "rating") {
      filtered.sort((a, b) => Number(b.rating) - Number(a.rating));
    }

    return filtered;
  }, [results, categoryFilter, priceFilter, sortBy]);

  const cartCount = cart.reduce((sum, item) => sum + item.quantity, 0);
  const cartTotal = cart.reduce(
    (sum, item) => sum + Number(item.price || 0) * item.quantity,
    0,
  );

  // Category browsing is separate from AI search.
  // It loads the complete catalog for the selected category.
  const browseCategory = async (category) => {
    if (loading) return;

    setLoading(true);
    setError("");
    setIntent(null);
    setBrowseMode(true);
    setActivePanel(null);
    setCategoryFilter("all");
    setPriceFilter("all");
    setSortBy("relevance");

    try {
      const response = await fetch(
        `${API_BASE}/products?category=${encodeURIComponent(category)}`
      );

      if (!response.ok) {
        throw new Error("Category service unavailable");
      }

      const data = await response.json();

      const products = Array.isArray(data)
        ? data
        : Array.isArray(data.products)
          ? data.products
          : Array.isArray(data.results)
            ? data.results
            : [];

      if (!products.length) {
        throw new Error("No products returned for this category");
      }

      setResults(products);
      setQuery("");
      window.setTimeout(() => scrollTo("results"), 80);
    } catch (err) {
      console.error("Category browse error:", err);
      setResults([]);
      setError("We couldn't load this category right now. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="navbar">
        <div className="nav-inner">
          <button className="brand" onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}>
            <span className="brand-mark">V</span>
            <span className="brand-name">VANTOS</span>
          </button>

          <nav className="nav-links" aria-label="Main navigation">
            <button onClick={() => scrollTo("discover")}>Discover</button>
            <button onClick={() => scrollTo("categories")}>Categories</button>
            <button onClick={() => scrollTo("ai-explained")}>How it works</button>
          </nav>

          <div className="nav-actions">
            <button
              className="icon-button"
              aria-label="Open wishlist"
              onClick={() => setActivePanel("wishlist")}
            >
              <Heart size={19} />
              {wishlist.length > 0 && <span className="count-badge">{wishlist.length}</span>}
            </button>

            <button
              className="bag-button"
              onClick={() => setActivePanel("bag")}
              aria-label="Open shopping bag"
            >
              <ShoppingBag size={18} />
              <span>Bag</span>
              {cartCount > 0 && <strong>{cartCount}</strong>}
            </button>
          </div>
        </div>
      </header>

      <main>
        <section className={`hero ${results.length ? "hero-compact" : ""}`} id="discover">
          <div className="hero-content">
            <div className="ai-badge">
              <Sparkles size={14} />
              AI-powered product discovery
            </div>

            <h1 className="hero-title">{displayText}<span className="typing-cursor" /></h1>

            <p className="hero-description">
              Tell VANTOS what you need in your own words. We understand intent,
              preferences and constraints to find relevant products.
            </p>

            <form
              className="search-form"
              onSubmit={(event) => {
                event.preventDefault();
                handleSearch();
              }}
            >
              <Search className="search-icon" size={21} />
              <input
                type="text"
                placeholder="Try “lightweight laptop for coding under ₹50,000”"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                aria-label="Search products"
              />
              <button className="search-button" type="submit" disabled={loading}>
                <ArrowRight size={20} />
              </button>
            </form>

            <div className="suggestions">
              <span>Try</span>
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  onClick={() => {
                    setQuery(suggestion);
                    setBrowseMode(false);
                  }}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="category-strip" id="categories">
          <div className="page-shell">
            <div className="section-topline">
              <div>
                <span className="eyebrow">SHOP BY CATEGORY</span>
                <h2>Start somewhere specific</h2>
              </div>
              <button
                className="text-button"
                onClick={() => {
                  setQuery("");
                  setResults([]);
                  setIntent(null);
                  setBrowseMode(false);
                  setError("");
                  setCategoryFilter("all");
                  setPriceFilter("all");
                  setSortBy("relevance");
                }}
              >
                Clear search <X size={14} />
              </button>
            </div>

            <div className="category-grid">
              {CATEGORY_CONFIG.map((category) => (
                <button
                  className="category-card"
                  key={category.value}
                  type="button"
                  onClick={() => browseCategory(category.value)}
                >
                  <span className="category-icon">{category.icon}</span>
                  <span>{category.label}</span>
                  <ChevronRight size={16} />
                </button>
              ))}
            </div>
          </div>
        </section>

        {loading && (
          <div className="loading-bar">
            <span />
            <p>Understanding your search and finding relevant products…</p>
          </div>
        )}

        {error && (
          <div className="page-shell">
            <div className="error-card">
              <div>
                <strong>Something went wrong</strong>
                <p>{error}</p>
              </div>
              <button onClick={() => handleSearch()} disabled={!query.trim()}>
                Try again
              </button>
            </div>
          </div>
        )}

        {results.length > 0 && !loading && (
          <section className="results-section page-shell" id="results">
            <div className="results-header">
              <div>
                <span className="eyebrow">
                  {browseMode ? "CATEGORY BROWSE" : "SEARCH RESULTS"}
                </span>
                <h2>
                  {browseMode
                    ? `${CATEGORY_CONFIG.find((item) => item.value === (results[0]?.category || ""))?.label || "Category"} products`
                    : "Products matched to your search"}
                </h2>
                <p>
                  {browseMode
                    ? `${visibleResults.length} products in this category`
                    : `${visibleResults.length} of ${results.length} AI-retrieved products shown`}
                </p>
              </div>

              <button className="filter-button" onClick={() => setShowFilters((value) => !value)}>
                <SlidersHorizontal size={17} />
                Filters & sort
                <ChevronDown size={15} className={showFilters ? "rotate" : ""} />
              </button>
            </div>

            <AnimatePresence>
              {showFilters && (
                <motion.div
                  className="filter-panel"
                  initial={{ opacity: 0, height: 0, y: -8 }}
                  animate={{ opacity: 1, height: "auto", y: 0 }}
                  exit={{ opacity: 0, height: 0, y: -8 }}
                >
                  <label>
                    Category
                    <select value={categoryFilter} onChange={(event) => setCategoryFilter(event.target.value)}>
                      <option value="all">All categories</option>
                      {CATEGORY_CONFIG.map((category) => (
                        <option value={category.value} key={category.value}>{category.label}</option>
                      ))}
                    </select>
                  </label>

                  <label>
                    Maximum price
                    <select value={priceFilter} onChange={(event) => setPriceFilter(event.target.value)}>
                      <option value="all">Any price</option>
                      <option value="10000">Under ₹10,000</option>
                      <option value="20000">Under ₹20,000</option>
                      <option value="50000">Under ₹50,000</option>
                      <option value="100000">Under ₹1,00,000</option>
                    </select>
                  </label>

                  <label>
                    Sort
                    <select value={sortBy} onChange={(event) => setSortBy(event.target.value)}>
                      <option value="relevance">AI relevance</option>
                      <option value="rating">Highest rated</option>
                      <option value="price-low">Price: low to high</option>
                      <option value="price-high">Price: high to low</option>
                    </select>
                  </label>

                  <button
                    className="reset-button"
                    onClick={() => {
                      setCategoryFilter("all");
                      setPriceFilter("all");
                      setSortBy("relevance");
                    }}
                  >
                    Reset
                  </button>
                </motion.div>
              )}
            </AnimatePresence>

            {intent && (
              <motion.div className="intent-card" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <div className="intent-icon"><Sparkles size={17} /></div>
                <div className="intent-content">
                  <span className="intent-label">VANTOS UNDERSTOOD</span>
                  <div className="intent-tags">
                    {intent.category && <span>{intent.category}</span>}
                    {intent.purpose && <span>{intent.purpose}</span>}
                    {intent.max_price && <span>Under {formatPrice(intent.max_price)}</span>}
                    {intent.preferences?.map((preference) => <span key={preference}>{preference}</span>)}
                  </div>
                </div>
              </motion.div>
            )}

            {visibleResults.length > 0 ? (
              <div className="product-grid">
                {visibleResults.map((product, index) => {
                  const wishlisted = wishlist.some((item) => item.product_id === product.product_id);

                  return (
                    <motion.article
                      className="product-card"
                      key={product.product_id}
                      initial={{ opacity: 0, y: 18 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.35, delay: index * 0.045 }}
                    >
                      <div className="product-image-wrap">
                        <button className="image-click" onClick={() => openProduct(product)} aria-label={`View ${product.product_name}`}>
                          <img src={getProductImage(product.category)} alt={product.product_name} />
                        </button>

                        {index === 0 && (
                          <span className="ai-pick-badge">
                            <Sparkles size={12} /> AI match
                          </span>
                        )}

                        <button
                          className={`wishlist-button ${wishlisted ? "active" : ""}`}
                          onClick={() => toggleWishlist(product)}
                          aria-label={wishlisted ? "Remove from wishlist" : "Add to wishlist"}
                        >
                          <Heart size={18} fill={wishlisted ? "currentColor" : "none"} />
                        </button>
                      </div>

                      <div className="product-info">
                        <span className="product-category">{product.category}</span>
                        <button className="product-title-button" onClick={() => openProduct(product)}>
                          {product.product_name}
                        </button>

                        <div className="product-rating">
                          <Star size={14} fill="currentColor" />
                          <strong>{product.rating}</strong>
                          <span>rating</span>
                        </div>

                        <div className="product-footer">
                          <strong className="product-price">{formatPrice(product.price)}</strong>
                          <button className="view-product" onClick={() => openProduct(product)}>
                            View details <ArrowRight size={15} />
                          </button>
                        </div>
                      </div>
                    </motion.article>
                  );
                })}
              </div>
            ) : (
              <div className="empty-results">
                <SlidersHorizontal size={22} />
                <h3>No products match these filters</h3>
                <p>Reset the filters to see the full AI result set.</p>
                <button onClick={() => {
                  setCategoryFilter("all");
                  setPriceFilter("all");
                  setSortBy("relevance");
                }}>
                  Reset filters
                </button>
              </div>
            )}
          </section>
        )}

        {!results.length && !loading && (
          <section className="discovery page-shell" id="ai-explained">
            <div className="discovery-intro">
              <span className="eyebrow">WHY VANTOS</span>
              <h2>Shopping that understands the request behind the words.</h2>
              <p>
                Instead of relying only on exact keywords, VANTOS combines semantic
                retrieval with AI intent understanding and product recommendations.
              </p>
            </div>

            <div className="feature-grid">
              <div className="feature-card">
                <span className="feature-number">01</span>
                <Sparkles size={20} />
                <h3>Understand</h3>
                <p>Extracts category, purpose, budget and preferences from natural language.</p>
              </div>
              <div className="feature-card">
                <span className="feature-number">02</span>
                <Search size={20} />
                <h3>Retrieve</h3>
                <p>Searches the product catalog using semantic similarity rather than keywords alone.</p>
              </div>
              <div className="feature-card">
                <span className="feature-number">03</span>
                <Zap size={20} />
                <h3>Recommend</h3>
                <p>Surfaces relevant alternatives, upgrades and complementary products.</p>
              </div>
            </div>
          </section>
        )}
      </main>

      <AnimatePresence>
        {selectedProduct && (
          <motion.div className="modal-overlay" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={closeProduct}>
            <motion.div
              className="product-modal"
              initial={{ opacity: 0, y: 20, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 20, scale: 0.98 }}
              onClick={(event) => event.stopPropagation()}
            >
              <button className="modal-close" onClick={closeProduct} aria-label="Close product details">
                <X size={20} />
              </button>

              <div className="modal-product">
                <div className="modal-image">
                  <img src={getProductImage(selectedProduct.category)} alt={selectedProduct.product_name} />
                  <span className="modal-category">{selectedProduct.category}</span>
                </div>

                <div className="modal-info">
                  <span className="eyebrow">PRODUCT DETAILS</span>
                  <h2>{selectedProduct.product_name}</h2>

                  <div className="modal-rating">
                    <Star size={16} fill="currentColor" />
                    <strong>{selectedProduct.rating}</strong>
                    <span>Customer rating</span>
                  </div>

                  <strong className="modal-price">{formatPrice(selectedProduct.price)}</strong>

                  <p className="modal-description">
                    {selectedProduct.description ||
                      "A product selected from the VANTOS catalog based on your search."}
                  </p>

                  {selectedProduct.features && (
                    <div className="feature-list">
                      {String(selectedProduct.features)
                        .split(",")
                        .slice(0, 4)
                        .map((feature) => (
                          <span key={feature}><Check size={13} />{feature.trim()}</span>
                        ))}
                    </div>
                  )}

                  <div className="modal-actions">
                    <button className="primary-action" onClick={() => addToBag(selectedProduct)}>
                      <ShoppingBag size={17} /> Add to bag
                    </button>
                    <button
                      className={`secondary-action ${
                        wishlist.some((item) => item.product_id === selectedProduct.product_id) ? "active" : ""
                      }`}
                      onClick={() => toggleWishlist(selectedProduct)}
                    >
                      <Heart
                        size={17}
                        fill={wishlist.some((item) => item.product_id === selectedProduct.product_id) ? "currentColor" : "none"}
                      />
                      Wishlist
                    </button>
                  </div>
                </div>
              </div>

              <div className="recommendations-area">
                <div className="recommendations-heading">
                  <div>
                    <span className="eyebrow">AI RECOMMENDATIONS</span>
                    <h3>Complete your purchase</h3>
                  </div>
                  <Sparkles size={18} />
                </div>

                {recommendationLoading && (
                  <div className="recommendation-loading">
                    <div className="loading-dot" />
                    Finding useful recommendations…
                  </div>
                )}

                {recommendationError && <div className="recommendation-error">{recommendationError}</div>}

                {!recommendationLoading && recommendations && (
                  <div className="recommendation-sections">
                    {[
                      {
                        key: "upsell",
                        title: "Upgrade options",
                        description: "Higher-value alternatives when more capability makes sense.",
                        icon: <Zap size={15} />,
                      },
                      {
                        key: "cross_sell",
                        title: "You may also need",
                        description: "Complementary products that fit this purchase.",
                        icon: <ShoppingBag size={15} />,
                      },
                    ].map((group) =>
                      recommendations[group.key]?.length ? (
                        <section className="recommendation-group" key={group.key}>
                          <div className="recommendation-group-header">
                            <span className="recommendation-icon">{group.icon}</span>
                            <div>
                              <h4>{group.title}</h4>
                              <p>{group.description}</p>
                            </div>
                          </div>

                          <div className="recommendation-grid">
                            {recommendations[group.key].map((product) => (
                              <button
                                className="recommendation-card"
                                key={product.product_id}
                                onClick={() => openProduct(product)}
                              >
                                <img src={getProductImage(product.category)} alt={product.product_name} />
                                <span>
                                  <small>{product.category}</small>
                                  <strong>{product.product_name}</strong>
                                  <em>{formatPrice(product.price)} · ★ {product.rating}</em>
                                </span>
                                <ChevronRight size={16} />
                              </button>
                            ))}
                          </div>
                        </section>
                      ) : null,
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {activePanel && (
          <motion.div className="drawer-overlay" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setActivePanel(null)}>
            <motion.aside
              className="side-drawer"
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              onClick={(event) => event.stopPropagation()}
            >
              <div className="drawer-header">
                <div>
                  <span className="eyebrow">{activePanel === "bag" ? "YOUR BAG" : "SAVED PRODUCTS"}</span>
                  <h2>{activePanel === "bag" ? `${cartCount} item${cartCount === 1 ? "" : "s"}` : `${wishlist.length} saved`}</h2>
                </div>
                <button className="icon-button" onClick={() => setActivePanel(null)}><X size={19} /></button>
              </div>

              {activePanel === "bag" ? (
                cart.length ? (
                  <>
                    <div className="drawer-list">
                      {cart.map((item) => (
                        <div className="drawer-item" key={item.product_id}>
                          <img src={getProductImage(item.category)} alt={item.product_name} />
                          <div>
                            <strong>{item.product_name}</strong>
                            <span>{formatPrice(item.price)}</span>
                            <div className="quantity-control">
                              <button onClick={() => changeQuantity(item.product_id, -1)}>−</button>
                              <b>{item.quantity}</b>
                              <button onClick={() => changeQuantity(item.product_id, 1)}>+</button>
                            </div>
                          </div>
                          <button className="remove-button" onClick={() => removeFromBag(item.product_id)}><X size={15} /></button>
                        </div>
                      ))}
                    </div>

                    <div className="drawer-total">
                      <span>Total</span>
                      <strong>{formatPrice(cartTotal)}</strong>
                    </div>
                    <button className="checkout-button" onClick={() => setToast("Demo checkout — no payment is connected.")}>
                      Continue to checkout <ArrowRight size={17} />
                    </button>
                  </>
                ) : (
                  <div className="drawer-empty">
                    <ShoppingBag size={30} />
                    <h3>Your bag is empty</h3>
                    <p>Add a product from the search results to see it here.</p>
                    <button onClick={() => {
                      setActivePanel(null);
                      scrollTo("discover");
                    }}>Continue shopping</button>
                  </div>
                )
              ) : (
                wishlist.length ? (
                  <div className="drawer-list">
                    {wishlist.map((item) => (
                      <button className="drawer-item wishlist-item" key={item.product_id} onClick={() => {
                        setActivePanel(null);
                        openProduct(item);
                      }}>
                        <img src={getProductImage(item.category)} alt={item.product_name} />
                        <div>
                          <strong>{item.product_name}</strong>
                          <span>{formatPrice(item.price)}</span>
                        </div>
                        <Heart size={16} fill="currentColor" />
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="drawer-empty">
                    <Heart size={30} />
                    <h3>Nothing saved yet</h3>
                    <p>Tap the heart on any product to keep it here.</p>
                    <button onClick={() => {
                      setActivePanel(null);
                      scrollTo("discover");
                    }}>Discover products</button>
                  </div>
                )
              )}
            </motion.aside>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {toast && (
          <motion.div className="toast" initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 15 }}>
            <Check size={15} /> {toast}
          </motion.div>
        )}
      </AnimatePresence>

      <footer className="footer">
        <div className="page-shell footer-inner">
          <strong>VANTOS</strong>
          <span>AI-powered product discovery</span>
          <span>© 2026</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
