import { Chart } from "@/components/ui/chart"
// Main JavaScript for Dangote Cement RAG Interface

class DangoteCementRAG {
  constructor() {
    this.chart = null
    this.currentChartType = "line"
    this.apiBase = "/api"

    this.initializeEventListeners()
    this.checkSystemHealth()
    this.initializeChart()
  }

  initializeEventListeners() {
    // Chart update button
    document.getElementById("updateChartBtn").addEventListener("click", () => {
      this.updateChart()
    })

    // Chart type buttons
    document.getElementById("lineChartBtn").addEventListener("click", () => {
      this.setChartType("line")
    })

    document.getElementById("barChartBtn").addEventListener("click", () => {
      this.setChartType("bar")
    })

    // Chat functionality
    document.getElementById("sendBtn").addEventListener("click", () => {
      this.sendChatMessage()
    })

    document.getElementById("chatInput").addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        this.sendChatMessage()
      }
    })

    // Ingest documents button
    document.getElementById("ingestBtn").addEventListener("click", () => {
      this.ingestDocuments()
    })

    // Metric selection change
    document.getElementById("metricSelect").addEventListener("change", () => {
      this.updateChart()
    })
  }

  async checkSystemHealth() {
    try {
      const response = await fetch(`${this.apiBase}/health`)
      const health = await response.json()

      const statusElement = document.getElementById("healthStatus")
      if (health.status === "healthy") {
        statusElement.innerHTML = `
                    <div class="w-3 h-3 bg-green-500 rounded-full"></div>
                    <span class="text-sm">System Ready</span>
                `
      } else {
        statusElement.innerHTML = `
                    <div class="w-3 h-3 bg-red-500 rounded-full"></div>
                    <span class="text-sm">System Error</span>
                `
      }
    } catch (error) {
      console.error("Health check failed:", error)
      const statusElement = document.getElementById("healthStatus")
      statusElement.innerHTML = `
                <div class="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <span class="text-sm">Connection Error</span>
            `
    }
  }

  initializeChart() {
    const ctx = document.getElementById("mainChart").getContext("2d")

    this.chart = new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: "Select a metric to view data",
          },
          legend: {
            display: true,
            position: "top",
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              callback: function (value) {
                return this.formatCurrency(value)
              }.bind(this),
            },
          },
        },
        interaction: {
          intersect: false,
          mode: "index",
        },
      },
    })
  }

  async updateChart() {
    const metric = document.getElementById("metricSelect").value
    const year = document.getElementById("yearSelect").value

    if (!metric) {
      this.showMessage("Please select a metric", "warning")
      return
    }

    this.showLoading("Updating chart...")

    try {
      let url = `${this.apiBase}/chart-data/${encodeURIComponent(metric)}`
      if (year) {
        url += `?years=${year}`
      }

      const response = await fetch(url)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const chartData = await response.json()

      // Update chart
      this.chart.data.labels = chartData.labels
      this.chart.data.datasets = chartData.datasets
      this.chart.options.plugins.title.text = `${metric} Over Time`
      this.chart.update()

      // Update data table
      await this.updateDataTable(metric, year)
    } catch (error) {
      console.error("Error updating chart:", error)
      this.showMessage("Error updating chart: " + error.message, "error")
    } finally {
      this.hideLoading()
    }
  }

  async updateDataTable(metric, selectedYear) {
    try {
      const response = await fetch(`${this.apiBase}/metric-data`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          metric: metric,
          years: selectedYear ? [Number.parseInt(selectedYear)] : null,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      const tableBody = document.getElementById("dataTableBody")

      if (!data.data || Object.keys(data.data).length === 0) {
        tableBody.innerHTML = `
                    <tr>
                        <td colspan="4" class="px-4 py-8 text-center text-gray-500">
                            No data available for ${metric}
                        </td>
                    </tr>
                `
        return
      }

      // Sort years and calculate growth rates
      const years = Object.keys(data.data).sort()
      let tableRows = ""

      years.forEach((year, index) => {
        const value = data.data[year]
        let growth = "-"

        if (index > 0) {
          const prevValue = data.data[years[index - 1]]
          if (prevValue && value) {
            const growthRate = ((value - prevValue) / prevValue) * 100
            growth = `${growthRate > 0 ? "+" : ""}${growthRate.toFixed(1)}%`
          }
        }

        const formattedValue = this.formatCurrency(value)
        const growthClass = growth.startsWith("+")
          ? "text-green-600"
          : growth.startsWith("-") && growth !== "-"
            ? "text-red-600"
            : "text-gray-500"

        tableRows += `
                    <tr class="hover:bg-gray-50">
                        <td class="px-4 py-2 text-sm font-medium text-gray-900">${year}</td>
                        <td class="px-4 py-2 text-sm text-gray-500">${metric}</td>
                        <td class="px-4 py-2 text-sm text-gray-900">${formattedValue}</td>
                        <td class="px-4 py-2 text-sm ${growthClass}">${growth}</td>
                    </tr>
                `
      })

      tableBody.innerHTML = tableRows
    } catch (error) {
      console.error("Error updating data table:", error)
      const tableBody = document.getElementById("dataTableBody")
      tableBody.innerHTML = `
                <tr>
                    <td colspan="4" class="px-4 py-8 text-center text-red-500">
                        Error loading data: ${error.message}
                    </td>
                </tr>
            `
    }
  }

  setChartType(type) {
    this.currentChartType = type

    // Update button styles
    document.getElementById("lineChartBtn").className =
      type === "line"
        ? "px-3 py-1 bg-blue-100 text-blue-600 rounded-md text-sm hover:bg-blue-200 transition-colors"
        : "px-3 py-1 bg-gray-100 text-gray-600 rounded-md text-sm hover:bg-gray-200 transition-colors"

    document.getElementById("barChartBtn").className =
      type === "bar"
        ? "px-3 py-1 bg-blue-100 text-blue-600 rounded-md text-sm hover:bg-blue-200 transition-colors"
        : "px-3 py-1 bg-gray-100 text-gray-600 rounded-md text-sm hover:bg-gray-200 transition-colors"

    // Update chart type
    if (this.chart) {
      this.chart.config.type = type
      this.chart.update()
    }
  }

  async sendChatMessage() {
    const input = document.getElementById("chatInput")
    const message = input.value.trim()

    if (!message) return

    // Add user message to chat
    this.addChatMessage(message, "user")
    input.value = ""

    // Show loading
    document.getElementById("chatLoading").classList.remove("hidden")

    try {
      const response = await fetch(`${this.apiBase}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: message,
          k: 5,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()

      // Add bot response to chat
      this.addChatMessage(result.answer, "bot", {
        confidence: result.confidence,
        queryType: result.query_type,
        sources: result.sources,
      })

      // Update last query in sidebar
      document.getElementById("lastQuery").textContent = message.substring(0, 20) + "..."
    } catch (error) {
      console.error("Error sending chat message:", error)
      this.addChatMessage("Sorry, I encountered an error processing your question. Please try again.", "bot")
    } finally {
      document.getElementById("chatLoading").classList.add("hidden")
    }
  }

  addChatMessage(message, sender, metadata = null) {
    const chatMessages = document.getElementById("chatMessages")
    const messageDiv = document.createElement("div")

    if (sender === "user") {
      messageDiv.className = "bg-blue-100 p-3 rounded-lg ml-8"
      messageDiv.innerHTML = `
                <div class="flex items-start space-x-2">
                    <i class="fas fa-user text-blue-600 mt-1"></i>
                    <p class="text-sm text-blue-800">${message}</p>
                </div>
            `
    } else {
      messageDiv.className = "bg-gray-100 p-3 rounded-lg mr-8"

      let confidenceBar = ""
      let queryTypeTag = ""
      let sourcesInfo = ""

      if (metadata) {
        const confidence = Math.round(metadata.confidence * 100)
        confidenceBar = `
                    <div class="mt-2 mb-2">
                        <div class="flex items-center space-x-2 text-xs text-gray-600">
                            <span>Confidence:</span>
                            <div class="flex-1 bg-gray-200 rounded-full h-2">
                                <div class="bg-green-500 h-2 rounded-full" style="width: ${confidence}%"></div>
                            </div>
                            <span>${confidence}%</span>
                        </div>
                    </div>
                `

        queryTypeTag = `
                    <span class="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full mr-2">
                        ${metadata.queryType}
                    </span>
                `

        if (metadata.sources && metadata.sources.length > 0) {
          sourcesInfo = `
                        <div class="mt-2 text-xs text-gray-600">
                            <i class="fas fa-link mr-1"></i>
                            ${metadata.sources.length} source(s) referenced
                        </div>
                    `
        }
      }

      messageDiv.innerHTML = `
                <div class="flex items-start space-x-2">
                    <i class="fas fa-robot text-gray-600 mt-1"></i>
                    <div class="flex-1">
                        <div class="mb-2">
                            ${queryTypeTag}
                        </div>
                        <p class="text-sm text-gray-800 whitespace-pre-wrap">${message}</p>
                        ${confidenceBar}
                        ${sourcesInfo}
                    </div>
                </div>
            `
    }

    chatMessages.appendChild(messageDiv)
    chatMessages.scrollTop = chatMessages.scrollHeight
  }

  async ingestDocuments() {
    this.showLoading("Ingesting documents...")

    try {
      const response = await fetch(`${this.apiBase}/ingest`, {
        method: "POST",
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()

      if (result.status === "success") {
        this.showMessage(`Successfully processed ${result.processed_files} files`, "success")
        document.getElementById("docCount").textContent = result.text_chunks
      } else {
        this.showMessage("Document ingestion failed", "error")
      }
    } catch (error) {
      console.error("Error ingesting documents:", error)
      this.showMessage("Error ingesting documents: " + error.message, "error")
    } finally {
      this.hideLoading()
    }
  }

  formatCurrency(value) {
    if (value === null || value === undefined) return "N/A"

    const num = Number.parseFloat(value)
    if (isNaN(num)) return "N/A"

    if (num >= 1e9) {
      return `₦${(num / 1e9).toFixed(2)}B`
    } else if (num >= 1e6) {
      return `₦${(num / 1e6).toFixed(2)}M`
    } else if (num >= 1e3) {
      return `₦${(num / 1e3).toFixed(2)}K`
    } else {
      return `₦${num.toFixed(2)}`
    }
  }

  showLoading(text = "Loading...") {
    const modal = document.getElementById("loadingModal")
    const loadingText = document.getElementById("loadingText")
    loadingText.textContent = text
    modal.classList.remove("hidden")
  }

  hideLoading() {
    const modal = document.getElementById("loadingModal")
    modal.classList.add("hidden")
  }

  showMessage(message, type = "info") {
    // Create a toast notification
    const toast = document.createElement("div")
    toast.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 max-w-sm ${
      type === "success"
        ? "bg-green-500 text-white"
        : type === "error"
          ? "bg-red-500 text-white"
          : type === "warning"
            ? "bg-yellow-500 text-white"
            : "bg-blue-500 text-white"
    }`

    toast.innerHTML = `
            <div class="flex items-center space-x-2">
                <i class="fas ${
                  type === "success"
                    ? "fa-check-circle"
                    : type === "error"
                      ? "fa-exclamation-circle"
                      : type === "warning"
                        ? "fa-exclamation-triangle"
                        : "fa-info-circle"
                }"></i>
                <span class="text-sm">${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" class="ml-2 text-white hover:text-gray-200">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `

    document.body.appendChild(toast)

    // Auto remove after 5 seconds
    setTimeout(() => {
      if (toast.parentElement) {
        toast.remove()
      }
    }, 5000)
  }
}

// Initialize the application when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  new DangoteCementRAG()
})
