package com.rakshakavach.app

import android.annotation.SuppressLint
import android.app.DownloadManager
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.webkit.CookieManager
import android.webkit.URLUtil
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private lateinit var tvStatus: TextView
    private var localUrl: String? = null
    private var filePathCallback: ValueCallback<Array<Uri>>? = null
    private val bgExecutor = Executors.newSingleThreadExecutor()

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
        tvStatus = findViewById(R.id.tvStatus)
        val btnReload = findViewById<Button>(R.id.btnReload)

        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = true
            loadWithOverviewMode = true
            useWideViewPort = true
            mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
        }

        webView.webViewClient = WebViewClient()

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

        btnReload.setOnClickListener {
            localUrl?.let { webView.loadUrl(it) } ?: startEmbeddedEngine()
        }

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

        showLoadingPage()
        startEmbeddedEngine()
    }

    private fun showLoadingPage() {
        val html = """
            <html><body style="font-family:sans-serif;padding:28px;background:#f8fafc;color:#0f172a;">
              <h2 style="color:#0b1f3a;">Raksha Kavach &mdash; Offline On-Device</h2>
              <p>Initializing embedded Python privacy firewall engine...</p>
            </body></html>
        """.trimIndent()
        webView.loadDataWithBaseURL(null, html, "text/html", "utf-8", null)
    }

    private fun startEmbeddedEngine() {
        bgExecutor.execute {
            try {
                if (!Python.isStarted()) {
                    Python.start(AndroidPlatform(applicationContext))
                }
                val py = Python.getInstance()
                val guiMod = py.getModule("gui")
                val url = guiMod.callAttr(
                    "start_background_server",
                    "127.0.0.1",
                    8573,
                    filesDir.absolutePath
                ).toString()

                runOnUiThread {
                    localUrl = url
                    tvStatus.text = "Offline On-Device ($url)"
                    webView.loadUrl(url)
                }
            } catch (t: Throwable) {
                runOnUiThread {
                    tvStatus.text = "Engine startup error"
                    val msg = t.message?.replace("<", "&lt;") ?: "Unknown error"
                    val html = """
                        <html><body style="font-family:sans-serif;padding:24px;background:#fff1f2;color:#881337;">
                          <h3>Embedded Python Startup Failed</h3>
                          <pre style="white-space:pre-wrap;">$msg</pre>
                        </body></html>
                    """.trimIndent()
                    webView.loadDataWithBaseURL(null, html, "text/html", "utf-8", null)
                }
            }
        }
    }

    override fun onDestroy() {
        bgExecutor.shutdownNow()
        super.onDestroy()
    }
}
