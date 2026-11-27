package com.rakshakavach.app

import android.annotation.SuppressLint
import android.app.AlertDialog
import android.app.DownloadManager
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.webkit.CookieManager
import android.webkit.URLUtil
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private lateinit var tvServerUrl: TextView
    private var filePathCallback: ValueCallback<Array<Uri>>? = null

    private val fileChooserLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val callback = filePathCallback ?: return@registerForActivityResult
        filePathCallback = null

        val data = result.data
        val uris = mutableListOf<Uri>()
        if (result.resultCode == RESULT_OK && data != null) {
            val clipData = data.clipData
            if (clipData != null) {
                for (i in 0 until clipData.itemCount) {
                    clipData.getItemAt(i)?.uri?.let { uris.add(it) }
                }
            } else {
                data.data?.let { uris.add(it) }
            }
        }
        callback.onReceiveValue(if (uris.isNotEmpty()) uris.toTypedArray() else null)
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        webView = findViewById(R.id.webView)
        tvServerUrl = findViewById(R.id.tvServerUrl)
        val btnServer = findViewById<Button>(R.id.btnServer)
        val btnReload = findViewById<Button>(R.id.btnReload)

        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = true
            loadWithOverviewMode = true
            useWideViewPort = true
            mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
        }

        webView.webViewClient = object : WebViewClient() {
            override fun onReceivedError(
                view: WebView,
                request: WebResourceRequest,
                error: WebResourceError
            ) {
                if (!request.isForMainFrame) return
                val url = getServerUrl()
                val html = """
                    <html><body style="font-family:sans-serif;padding:24px;background:#f8fafc;color:#0f172a;">
                      <h2 style="color:#0b1f3a;">Raksha Kavach Server Not Reachable</h2>
                      <p>Could not connect to <code>$url</code>.</p>
                      <ul>
                        <li>Start the Python GUI on your Mac:<br>
                            <code>.venv/bin/python gui.py --host 0.0.0.0 --port 8573</code></li>
                        <li>For USB debugging: run <code>adb reverse tcp:8573 tcp:8573</code> and use <code>http://127.0.0.1:8573/</code>.</li>
                        <li>For Android Emulator: tap <b>Server</b> and choose <code>http://10.0.2.2:8573/</code>.</li>
                        <li>For Wi-Fi phone: tap <b>Server</b> and enter your Mac's LAN IP, e.g. <code>http://192.168.1.10:8573/</code>.</li>
                      </ul>
                    </body></html>
                """.trimIndent()
                view.loadDataWithBaseURL(null, html, "text/html", "utf-8", null)
            }
        }

        webView.webChromeClient = object : WebChromeClient() {
            override fun onShowFileChooser(
                view: WebView,
                callback: ValueCallback<Array<Uri>>,
                params: FileChooserParams
            ): Boolean {
                filePathCallback?.onReceiveValue(null)
                filePathCallback = callback
                val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "*/*"
                    putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
                }
                fileChooserLauncher.launch(Intent.createChooser(intent, "Select clinical files"))
                return true
            }
        }

        webView.setDownloadListener { url, userAgent, contentDisposition, mimetype, _ ->
            if (!url.startsWith("http://") && !url.startsWith("https://")) {
                Toast.makeText(this, "Unsupported download URL scheme", Toast.LENGTH_SHORT).show()
                return@setDownloadListener
            }
            val request = DownloadManager.Request(Uri.parse(url)).apply {
                setMimeType(mimetype)
                addRequestHeader("Cookie", CookieManager.getInstance().getCookie(url))
                addRequestHeader("User-Agent", userAgent)
                val filename = URLUtil.guessFileName(url, contentDisposition, mimetype)
                setTitle(filename)
                setDescription("Downloading sanitized bundle")
                setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, filename)
            }
            val dm = getSystemService(DOWNLOAD_SERVICE) as DownloadManager
            dm.enqueue(request)
            Toast.makeText(this, "Downloading sanitized output...", Toast.LENGTH_SHORT).show()
        }

        btnServer.setOnClickListener { showServerDialog() }
        btnReload.setOnClickListener { loadCurrentUrl() }

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (webView.canGoBack()) {
                    webView.goBack()
                } else {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                }
            }
        })

        loadCurrentUrl()
    }

    private fun getServerUrl(): String {
        val prefs = getSharedPreferences("raksha_prefs", MODE_PRIVATE)
        return prefs.getString("server_url", "http://127.0.0.1:8573/") ?: "http://127.0.0.1:8573/"
    }

    private fun saveServerUrl(raw: String) {
        var cleaned = raw.trim()
        if (!cleaned.startsWith("http://") && !cleaned.startsWith("https://")) {
            cleaned = "http://$cleaned"
        }
        if (!cleaned.endsWith("/")) {
            cleaned += "/"
        }
        getSharedPreferences("raksha_prefs", MODE_PRIVATE)
            .edit()
            .putString("server_url", cleaned)
            .apply()
        loadCurrentUrl()
    }

    private fun loadCurrentUrl() {
        val url = getServerUrl()
        tvServerUrl.text = url
        webView.loadUrl(url)
    }

    private fun showServerDialog() {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 24, 48, 8)
        }
        val input = EditText(this).apply {
            setText(getServerUrl())
            hint = "http://192.168.x.x:8573/"
            setSingleLine(true)
        }
        container.addView(input)

        AlertDialog.Builder(this)
            .setTitle("Raksha Kavach Backend URL")
            .setMessage("Use 127.0.0.1 (USB adb reverse), 10.0.2.2 (Emulator), or your Mac's Wi-Fi IP.")
            .setView(container)
            .setPositiveButton("Connect") { _, _ -> saveServerUrl(input.text.toString()) }
            .setNeutralButton("Emulator (10.0.2.2)") { _, _ -> saveServerUrl("http://10.0.2.2:8573/") }
            .setNegativeButton("USB (127.0.0.1)") { _, _ -> saveServerUrl("http://127.0.0.1:8573/") }
            .show()
    }
}
