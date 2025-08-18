document.addEventListener('DOMContentLoaded', async function() {
  const elements = {
    website: document.getElementById('website'),
    searchQuery: document.getElementById('searchQuery'),
    startPage: document.getElementById('startPage'),
    endPage: document.getElementById('endPage'),
    scrapeDetails: document.getElementById('scrapeDetails'),
    startBtn: document.getElementById('startBtn'),
    stopBtn: document.getElementById('stopBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    clearBtn: document.getElementById('clearBtn'),
    status: document.getElementById('status'),
    progressBar: document.getElementById('progressBar'),
    pagesScraped: document.getElementById('pagesScraped'),
    productsFound: document.getElementById('productsFound')
  };

  let scrapingActive = false;
  let scrapedData = [];

  // Load saved data
  await loadStoredData();

  // Event listeners
  elements.startBtn.addEventListener('click', startScraping);
  elements.stopBtn.addEventListener('click', stopScraping);
  elements.downloadBtn.addEventListener('click', downloadCSV);
  elements.clearBtn.addEventListener('click', clearData);

  // Listen for updates from background script
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'scraping_progress') {
      updateProgress(message.data);
    } else if (message.type === 'scraping_complete') {
      scrapingComplete(message.data);
    } else if (message.type === 'scraping_error') {
      showError(message.error);
    }
  });

  async function startScraping() {
    const config = {
      website: elements.website.value,
      searchQuery: elements.searchQuery.value.trim(),
      startPage: parseInt(elements.startPage.value),
      endPage: parseInt(elements.endPage.value),
      scrapeDetails: elements.scrapeDetails.value === 'true'
    };

    if (!config.searchQuery) {
      showError('Please enter a search query');
      return;
    }

    if (config.startPage > config.endPage) {
      showError('Start page cannot be greater than end page');
      return;
    }

    scrapingActive = true;
    elements.startBtn.style.display = 'none';
    elements.stopBtn.style.display = 'block';
    elements.status.className = 'status running';
    elements.status.textContent = 'Starting scraper...';
    document.querySelector('.progress').style.display = 'block';

    // Send message to background script to start scraping
    chrome.runtime.sendMessage({
      action: 'start_scraping',
      config: config
    });
  }

  async function stopScraping() {
    scrapingActive = false;
    chrome.runtime.sendMessage({ action: 'stop_scraping' });
    
    elements.startBtn.style.display = 'block';
    elements.stopBtn.style.display = 'none';
    elements.status.className = 'status idle';
    elements.status.textContent = 'Scraping stopped';
    document.querySelector('.progress').style.display = 'none';
  }

  function updateProgress(data) {
    const { currentPage, totalPages, productsCount, pagesCount } = data;
    
    const progress = (currentPage / totalPages) * 100;
    elements.progressBar.style.width = `${progress}%`;
    elements.progressBar.textContent = `${Math.round(progress)}%`;
    
    elements.pagesScraped.textContent = pagesCount;
    elements.productsFound.textContent = productsCount;
    
    elements.status.textContent = `Scraping page ${currentPage} of ${totalPages}...`;
  }

  async function scrapingComplete(data) {
    scrapingActive = false;
    scrapedData = data.products || [];
    
    elements.startBtn.style.display = 'block';
    elements.stopBtn.style.display = 'none';
    elements.status.className = 'status complete';
    elements.status.textContent = `✅ Complete! Found ${scrapedData.length} products`;
    
    elements.progressBar.style.width = '100%';
    elements.progressBar.textContent = '100%';
    
    if (scrapedData.length > 0) {
      elements.downloadBtn.style.display = 'block';
    }

    // Store data
    await chrome.storage.local.set({ 
      scrapedProducts: scrapedData,
      lastScrapeTime: Date.now()
    });
  }

  function showError(error) {
    elements.status.className = 'status error';
    elements.status.textContent = `❌ Error: ${error}`;
    
    elements.startBtn.style.display = 'block';
    elements.stopBtn.style.display = 'none';
    document.querySelector('.progress').style.display = 'none';
  }

  async function downloadCSV() {
    if (scrapedData.length === 0) {
      showError('No data to download');
      return;
    }

    const csvContent = convertToCSV(scrapedData);
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = `${elements.website.value}_products_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
  }

  async function clearData() {
    await chrome.storage.local.clear();
    scrapedData = [];
    
    elements.pagesScraped.textContent = '0';
    elements.productsFound.textContent = '0';
    elements.downloadBtn.style.display = 'none';
    elements.status.className = 'status idle';
    elements.status.textContent = 'Data cleared - Ready to scrape';
    elements.progressBar.style.width = '0%';
    document.querySelector('.progress').style.display = 'none';
  }

  async function loadStoredData() {
    const result = await chrome.storage.local.get(['scrapedProducts', 'lastScrapeTime']);
    
    if (result.scrapedProducts && result.scrapedProducts.length > 0) {
      scrapedData = result.scrapedProducts;
      elements.productsFound.textContent = scrapedData.length;
      elements.downloadBtn.style.display = 'block';
      
      const lastScrape = new Date(result.lastScrapeTime);
      elements.status.className = 'status complete';
      elements.status.textContent = `Last scraped: ${lastScrape.toLocaleString()}`;
    }
  }

  function convertToCSV(data) {
    if (data.length === 0) return '';
    
    const headers = Object.keys(data[0]);
    const csvRows = [headers.join(',')];
    
    for (const row of data) {
      const values = headers.map(header => {
        const value = row[header] || '';
        return `"${String(value).replace(/"/g, '""')}"`;
      });
      csvRows.push(values.join(','));
    }
    
    return csvRows.join('\n');
  }
});