// background.js
(() => {
  let scrapingState = {
    active: false,
    config: null,
    currentPage: 0,
    totalPages: 0,
    products: [],
    visitedUrls: new Set()
  };

  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'start_scraping') {
      startScrapingProcess(request.config);
    } else if (request.action === 'stop_scraping') {
      stopScrapingProcess();
    } else if (request.action === 'page_scraped') {
      handlePageScraped(request.data);
    } else if (request.action === 'product_detail_scraped') {
      handleProductDetailScraped(request.data);
    } else if (request.action === 'scraping_error') {
      handleScrapingError(request.error);
    }
  });

  async function startScrapingProcess(config) {
    scrapingState = {
      active: true,
      config: config,
      currentPage: config.startPage,
      totalPages: config.endPage - config.startPage + 1,
      products: [],
      visitedUrls: new Set(),
      detailQueue: [],
      processingDetails: false
    };
    console.log('Starting scraping process:', config);
    await scrapeNextPage();
  }

  function stopScrapingProcess() {
    scrapingState.active = false;
    console.log('Scraping process stopped');
  }

  async function scrapeNextPage() {
    if (!scrapingState.active || scrapingState.currentPage > scrapingState.config.endPage) {
      if (scrapingState.config.scrapeDetails && scrapingState.detailQueue.length > 0) {
        await processDetailQueue();
      } else {
        completeScraping();
      }
      return;
    }

    const url = buildSearchUrl(scrapingState.config, scrapingState.currentPage);
    console.log(`Scraping page ${scrapingState.currentPage}: ${url}`);

    try {
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      const currentTab = tabs[0];
      await chrome.tabs.update(currentTab.id, { url: url });
      chrome.runtime.sendMessage({
        type: 'scraping_progress',
        data: {
          currentPage: scrapingState.currentPage,
          totalPages: scrapingState.totalPages,
          productsCount: scrapingState.products.length,
          pagesCount: scrapingState.currentPage - scrapingState.config.startPage
        }
      });
    } catch (error) {
      handleScrapingError(`Failed to navigate to page ${scrapingState.currentPage}: ${error.message}`);
    }
  }

  function buildSearchUrl(config, page) {
    const query = encodeURIComponent(config.searchQuery);
    if (config.website === 'flipkart') {
      return `https://www.flipkart.com/search?q=${query}&page=${page}`;
    } else if (config.website === 'amazon') {
      return `https://www.amazon.in/s?k=${query}&page=${page}`;
    }
    throw new Error('Unsupported website');
  }

  function handlePageScraped(data) {
    if (!scrapingState.active) return;
    console.log(`Page ${scrapingState.currentPage} scraped:`, data);
    if (data.products && data.products.length > 0) {
      scrapingState.products.push(...data.products);
      if (scrapingState.config.scrapeDetails) {
        const detailUrls = data.products
          .filter(p => p.productUrl && !scrapingState.visitedUrls.has(p.productUrl))
          .map(p => p.productUrl);
        scrapingState.detailQueue.push(...detailUrls);
        detailUrls.forEach(url => scrapingState.visitedUrls.add(url));
      }
    }
    scrapingState.currentPage++;
    setTimeout(() => {
      scrapeNextPage();
    }, 2000);
  }

  async function processDetailQueue() {
    if (!scrapingState.active || scrapingState.detailQueue.length === 0) {
      completeScraping();
      return;
    }
    scrapingState.processingDetails = true;
    for (let i = 0; i < scrapingState.detailQueue.length && scrapingState.active; i++) {
      const productUrl = scrapingState.detailQueue[i];
      try {
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        const currentTab = tabs[0];
        await chrome.tabs.update(currentTab.id, { url: productUrl });
        await new Promise(resolve => {
          const timeout = setTimeout(resolve, 10000);
          const listener = (message) => {
            if (message.action === 'product_detail_scraped') {
              clearTimeout(timeout);
              chrome.runtime.onMessage.removeListener(listener);
              resolve();
            }
          };
          chrome.runtime.onMessage.addListener(listener);
        });
        chrome.runtime.sendMessage({
          type: 'scraping_progress',
          data: {
            currentPage: `Detail ${i + 1}/${scrapingState.detailQueue.length}`,
            totalPages: scrapingState.detailQueue.length,
            productsCount: scrapingState.products.length,
            pagesCount: scrapingState.currentPage - scrapingState.config.startPage
          }
        });
        await new Promise(resolve => setTimeout(resolve, 3000));
      } catch (error) {
        console.error(`Failed to scrape detail page ${productUrl}:`, error);
      }
    }
    completeScraping();
  }

  function handleProductDetailScraped(data) {
    if (!scrapingState.active || !data.productUrl) return;
    const productIndex = scrapingState.products.findIndex(p => p.productUrl === data.productUrl);
    if (productIndex !== -1) {
      scrapingState.products[productIndex] = {
        ...scrapingState.products[productIndex],
        ...data.details
      };
      console.log('Updated product with details:', scrapingState.products[productIndex]);
    }
  }

  function handleScrapingError(error) {
    console.error('Scraping error:', error);
    scrapingState.active = false;
    chrome.runtime.sendMessage({
      type: 'scraping_error',
      error: error
    });
  }

  function completeScraping() {
    console.log('Scraping completed. Total products:', scrapingState.products.length);
    chrome.runtime.sendMessage({
      type: 'scraping_complete',
      data: {
        products: scrapingState.products,
        totalPages: scrapingState.currentPage - scrapingState.config.startPage,
        totalProducts: scrapingState.products.length
      }
    });
    scrapingState.active = false;
  }
})();
