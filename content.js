// Content script - runs on Flipkart and Amazon pages
console.log('Product scraper content script loaded');

// Wait for page to load then start scraping
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeScraper);
} else {
  initializeScraper();
}

function initializeScraper() {
  // Wait a bit more for dynamic content to load
  setTimeout(() => {
    const currentUrl = window.location.href;
    
    if (isSearchPage(currentUrl)) {
      scrapeSearchPage();
    } else if (isProductPage(currentUrl)) {
      scrapeProductPage();
    }
  }, 3000);
}

function isSearchPage(url) {
  return url.includes('search') || url.includes('/s?k=');
}

function isProductPage(url) {
  return url.includes('/p/') || url.includes('/dp/') || url.includes('/gp/product/');
}

function scrapeSearchPage() {
  console.log('Scraping search page...');
  
  const website = window.location.hostname.includes('flipkart') ? 'flipkart' : 'amazon';
  let products = [];
  
  if (website === 'flipkart') {
    products = scrapeFlipkartSearchPage();
  } else if (website === 'amazon') {
    products = scrapeAmazonSearchPage();
  }
  
  console.log(`Found ${products.length} products on search page`);
  
  // Send scraped data to background script
  chrome.runtime.sendMessage({
    action: 'page_scraped',
    data: {
      products: products,
      url: window.location.href
    }
  });
}

function scrapeFlipkartSearchPage() {
  const products = [];
  
  // Updated selectors based on current Flipkart structure
  const productContainers = document.querySelectorAll('[data-id]');
  
  productContainers.forEach((container, index) => {
    try {
      const product = {};
      
      // Product name
      const nameElement = container.querySelector('div[class*="KzDlHZ"], div[class*="_4rR01T"], a[class*="IRpwTa"]');
      product.name = nameElement ? nameElement.textContent.trim() : `Product ${index + 1}`;
      
      // Current price
      const priceElement = container.querySelector('div[class*="Nx9bqj"], div[class*="_30jeq3"], div[class*="_1_WHN1"]');
      product.currentPrice = priceElement ? priceElement.textContent.trim() : 'N/A';
      
      // Previous price
      const originalPriceElement = container.querySelector('div[class*="yRaY8j"], div[class*="_3I9_wc"], div[class*="ZYYwLA"]');
      product.originalPrice = originalPriceElement ? originalPriceElement.textContent.trim() : 'N/A';
      
      // Description/features
      const descElement = container.querySelector('div[class*="_6NESgJ"], ul[class*="_1xgFaf"], div[class*="fMghEO"]');
      product.description = descElement ? descElement.textContent.trim() : 'No description';
      
      // Rating
      const ratingElement = container.querySelector('div[class*="XQDdHH"], span[class*="_2_R_DZ"], div[class*="_3LWZlK"]');
      product.rating = ratingElement ? ratingElement.textContent.trim() : 'No rating';
      
      // Reviews count
      const reviewElement = container.querySelector('span[class*="Wphh3N"], span[class*="_2_R_DZ"], span[class*="_2V1EHk"]');
      product.reviews = reviewElement ? reviewElement.textContent.trim() : '0';
      
      // Image
      const imageElement = container.querySelector('img');
      product.image = imageElement ? (imageElement.src || imageElement.getAttribute('data-src')) : 'No image';
      
      // Product URL for detailed scraping
      const linkElement = container.querySelector('a[href*="/p/"]');
      if (linkElement) {
        const href = linkElement.getAttribute('href');
        product.productUrl = href.startsWith('http') ? href : `https://www.flipkart.com${href}`;
      }
      
      // Additional Flipkart specific data
      const deliveryElement = container.querySelector('div[class*="_2Tpdn3"], div[class*="_3tcehv"]');
      product.delivery = deliveryElement ? deliveryElement.textContent.trim() : 'N/A';
      
      const offersElement = container.querySelector('div[class*="_1xgFaf"], div[class*="_16FRp0"]');
      product.offers = offersElement ? offersElement.textContent.trim() : 'No offers';
      
      products.push(product);
      
    } catch (error) {
      console.error('Error scraping product:', error);
    }
  });
  
  return products;
}

function scrapeAmazonSearchPage() {
  const products = [];
  
  // Amazon product containers
  const productContainers = document.querySelectorAll('[data-component-type="s-search-result"]');
  
  productContainers.forEach((container, index) => {
    try {
      const product = {};
      
      // Product name
      const nameElement = container.querySelector('h2 a span, h2 span');
      product.name = nameElement ? nameElement.textContent.trim() : `Product ${index + 1}`;
      
      // Current price
      const priceElement = container.querySelector('.a-price-whole, .a-price .a-offscreen');
      product.currentPrice = priceElement ? priceElement.textContent.trim() : 'N/A';
      
      // Original price
      const originalPriceElement = container.querySelector('.a-price.a-text-price .a-offscreen, .a-price[data-a-strike="true"] .a-offscreen');
      product.originalPrice = originalPriceElement ? originalPriceElement.textContent.trim() : 'N/A';
      
      // Rating
      const ratingElement = container.querySelector('.a-icon-alt, [aria-label*="stars"]');
      product.rating = ratingElement ? ratingElement.getAttribute('aria-label') || ratingElement.textContent.trim() : 'No rating';
      
      // Reviews count
      const reviewElement = container.querySelector('a[href*="#customerReviews"] span, .a-size-base');
      product.reviews = reviewElement ? reviewElement.textContent.trim() : '0';
      
      // Image
      const imageElement = container.querySelector('.s-image, .a-dynamic-image');
      product.image = imageElement ? imageElement.src : 'No image';
      
      // Product URL
      const linkElement = container.querySelector('h2 a, .a-link-normal');
      if (linkElement) {
        const href = linkElement.getAttribute('href');
        product.productUrl = href.startsWith('http') ? href : `https://www.amazon.in${href}`;
      }
      
      // Delivery info
      const deliveryElement = container.querySelector('[aria-label*="FREE delivery"], .a-color-base:contains("FREE")');
      product.delivery = deliveryElement ? deliveryElement.textContent.trim() : 'N/A';
      
      // Prime badge
      const primeElement = container.querySelector('.a-icon-prime, [aria-label*="Prime"]');
      product.prime = primeElement ? 'Prime' : 'No';
      
      // Sponsored
      const sponsoredElement = container.querySelector('.s-sponsored-label-text');
      product.sponsored = sponsoredElement ? 'Yes' : 'No';
      
      products.push(product);
      
    } catch (error) {
      console.error('Error scraping Amazon product:', error);
    }
  });
  
  return products;
}

function scrapeProductPage() {
  console.log('Scraping product detail page...');
  
  const website = window.location.hostname.includes('flipkart') ? 'flipkart' : 'amazon';
  let productDetails = {};
  
  if (website === 'flipkart') {
    productDetails = scrapeFlipkartProductPage();
  } else if (website === 'amazon') {
    productDetails = scrapeAmazonProductPage();
  }
  
  console.log('Product details scraped:', productDetails);
  
  // Send detailed product data to background script
  chrome.runtime.sendMessage({
    action: 'product_detail_scraped',
    data: {
      productUrl: window.location.href,
      details: productDetails
    }
  });
}

function scrapeFlipkartProductPage() {
  const details = {};
  
  try {
    // Available colors/variants
    const colorElements = document.querySelectorAll('.swatch-container img, ._53J4C- img, .color-variant img');
    details.availableColors = Array.from(colorElements).map(img => 
      img.getAttribute('alt') || img.getAttribute('title') || 'Color variant'
    );
    
    // Additional images
    const imageElements = document.querySelectorAll('._2r_T1I img, ._396cs4 img, .q6DClP img');
    details.allImages = Array.from(imageElements).map(img => img.src).filter(src => src && !src.includes('data:'));
    
    // Seller information
    const sellerElement = document.querySelector('#sellerName span, .sdMJKL, ._1fkiLI');
    details.seller = sellerElement ? sellerElement.textContent.trim() : 'N/A';
    
    // Seller rating
    const sellerRatingElement = document.querySelector('.XQDdHH Gn+jFg span, ._3Ur5oX, .hGSR34');
    details.sellerRating = sellerRatingElement ? sellerRatingElement.textContent.trim() : 'N/A';
    
    // Detailed rating breakdown
    const ratingBars = document.querySelectorAll('._1i0wk8 ._3LWZlK, .row ._3LWZlK');
    details.ratingBreakdown = Array.from(ratingBars).map(bar => {
      const stars = bar.querySelector('._3LWZlK')?.textContent.trim() || '';
      const count = bar.querySelector('._2_R_DZ')?.textContent.trim() || '';
      return `${stars}: ${count}`;
    });
    
    // Specifications
    const specRows = document.querySelectorAll('._1s_Smc tr, .spec-table tr, ._2418kt tr');
    details.specifications = {};
    specRows.forEach(row => {
      const key = row.querySelector('td:first-child')?.textContent.trim();
      const value = row.querySelector('td:last-child')?.textContent.trim();
      if (key && value) {
        details.specifications[key] = value;
      }
    });
    
    // Highlights/Key features
    const highlights = document.querySelectorAll('.yN+eNk li, ._21Ahn- li, .highlight li');
    details.highlights = Array.from(highlights).map(li => li.textContent.trim());
    
    // EMI options
    const emiElement = document.querySelector('.zl_2WC, ._3pAKcx, .emi-info');
    details.emiOptions = emiElement ? emiElement.textContent.trim() : 'N/A';
    
    // Stock status
    const stockElement = document.querySelector('._16FRp0, .stock-info, ._3xgqrz');
    details.stockStatus = stockElement ? stockElement.textContent.trim() : 'In Stock';
    
    // Offers
    const offerElements = document.querySelectorAll('._3j4Zjp li, .offer-item li, ._16FRp0 li');
    details.offers = Array.from(offerElements).map(li => li.textContent.trim());
    
  } catch (error) {
    console.error('Error scraping Flipkart product details:', error);
  }
  
  return details;
}

function scrapeAmazonProductPage() {
  const details = {};
  
  try {
    // Available colors/variants
    const colorElements = document.querySelectorAll('#variation_color_name li img, .swatches li img');
    details.availableColors = Array.from(colorElements).map(img => 
      img.getAttribute('alt') || img.getAttribute('title') || 'Color variant'
    );
    
    // Size variants
    const sizeElements = document.querySelectorAll('#variation_size_name option, .size-variants option');
    details.availableSizes = Array.from(sizeElements)
      .filter(option => option.value && option.value !== '')
      .map(option => option.textContent.trim());
    
    // Additional images
    const imageElements = document.querySelectorAll('#altImages img, .image-wrapper img');
    details.allImages = Array.from(imageElements).map(img => img.src).filter(src => src && !src.includes('data:'));
    
    // Seller information
    const sellerElement = document.querySelector('#sellerProfileTriggerId, .offer-seller-name, #merchant-info');
    details.seller = sellerElement ? sellerElement.textContent.trim() : 'Amazon';
    
    // Detailed rating breakdown
    const ratingTable = document.querySelector('#histogramTable, .histogram');
    if (ratingTable) {
      const ratingRows = ratingTable.querySelectorAll('tr');
      details.ratingBreakdown = Array.from(ratingRows).map(row => {
        const stars = row.querySelector('.a-size-base')?.textContent.trim() || '';
        const percentage = row.querySelector('.a-right .a-size-base')?.textContent.trim() || '';
        return `${stars}: ${percentage}`;
      });
    }
    
    // Product details/specifications
    const specSections = document.querySelectorAll('#feature-bullets ul li, #productDetails_detailBullets_sections1 tr');
    details.specifications = {};
    specSections.forEach(item => {
      if (item.tagName === 'LI') {
        const text = item.textContent.trim();
        if (text && !text.includes('Make sure') && !text.includes('Report an issue')) {
          details.specifications[`Feature ${Object.keys(details.specifications).length + 1}`] = text;
        }
      } else if (item.tagName === 'TR') {
        const key = item.querySelector('td:first-child')?.textContent.trim();
        const value = item.querySelector('td:last-child')?.textContent.trim();
        if (key && value) {
          details.specifications[key] = value;
        }
      }
    });
    
    // Product description
    const descElement = document.querySelector('#feature-bullets, #product-description');
    details.fullDescription = descElement ? descElement.textContent.trim() : 'N/A';
    
    // Brand
    const brandElement = document.querySelector('#bylineInfo, .author .a-link-normal');
    details.brand = brandElement ? brandElement.textContent.trim() : 'N/A';
    
    // Availability
    const availabilityElement = document.querySelector('#availability span, #merchant-info');
    details.availability = availabilityElement ? availabilityElement.textContent.trim() : 'In Stock';
    
    // Delivery info
    const deliveryElement = document.querySelector('#mir-layout-DELIVERY_BLOCK, .delivery-info');
    details.deliveryInfo = deliveryElement ? deliveryElement.textContent.trim() : 'N/A';
    
    // Best seller rank
    const rankElement = document.querySelector('#SalesRank, .rank');
    details.salesRank = rankElement ? rankElement.textContent.trim() : 'N/A';
    
    // Customer questions
    const questionsElement = document.querySelector('#ask-dp-search_feature_div');
    details.hasQuestions = questionsElement ? 'Yes' : 'No';
    
  } catch (error) {
    console.error('Error scraping Amazon product details:', error);
  }
  
  return details;
}

// Helper function to wait for elements to load
function waitForElement(selector, timeout = 10000) {
  return new Promise((resolve, reject) => {
    const element = document.querySelector(selector);
    if (element) {
      resolve(element);
      return;
    }

    const observer = new MutationObserver((mutations) => {
      const element = document.querySelector(selector);
      if (element) {
        observer.disconnect();
        resolve(element);
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });

    setTimeout(() => {
      observer.disconnect();
      reject(new Error('Element not found within timeout'));
    }, timeout);
  });
}

// Handle dynamic content loading
function handleDynamicContent() {
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
        // Check if new product elements were added
        const hasNewProducts = Array.from(mutation.addedNodes).some(node => 
          node.nodeType === Node.ELEMENT_NODE && 
          (node.querySelector('[data-id]') || node.querySelector('[data-component-type="s-search-result"]'))
        );
        
        if (hasNewProducts) {
          console.log('New products detected, re-scraping...');
          setTimeout(() => {
            if (isSearchPage(window.location.href)) {
              scrapeSearchPage();
            }
          }, 2000);
        }
      }
    });
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
}